import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.icr_score import ICRScore


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    attn_implementation="eager",
).to("cuda")

model.eval()


def extract_icr_features(prompt, max_new_tokens=10):
    """
    Given a prompt:

    1. Generate an answer token-by-token.
    2. Save hidden states and attentions.
    3. Calculate ICR scores.
    4. Average ICR across generated tokens.
    5. Return:
       - generated text
       - one ICR value per transformer layer
    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    hidden_states = []
    attentions = []
    generated_tokens = []

    with torch.no_grad():

        # -----------------------------
        # Process the original prompt
        # -----------------------------
        outputs = model(
            **inputs,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

        hidden_states.append(
            tuple(x.detach().cpu() for x in outputs.hidden_states)
        )

        attentions.append(
            tuple(x.detach().cpu() for x in outputs.attentions)
        )

        past_key_values = outputs.past_key_values

        # First token predicted after the prompt
        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1
        ).unsqueeze(0)

        # -----------------------------
        # Generate response
        # -----------------------------
        for _ in range(max_new_tokens):

            generated_tokens.append(next_token.item())

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

            hidden_states.append(
                tuple(x.detach().cpu() for x in outputs.hidden_states)
            )

            attentions.append(
                tuple(x.detach().cpu() for x in outputs.attentions)
            )

            past_key_values = outputs.past_key_values

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1
            ).unsqueeze(0)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    # ---------------------------------------
    # Prepare the tensors for ICR computation
    # ---------------------------------------

    input_len = hidden_states[0][0].shape[1]

    core_positions = {
        "user_prompt_start": 0,
        "user_prompt_end": input_len,
        "response_start": input_len,
    }

    # Convert to FP32 because ICR math was
    # numerically unstable in FP16.
    hidden_states = [
        tuple(x.cuda().float() for x in step)
        for step in hidden_states
    ]

    attentions = [
        tuple(x.cuda().float() for x in step)
        for step in attentions
    ]

    # -----------------------------
    # Calculate ICR
    # -----------------------------

    icr = ICRScore(
        hidden_states=hidden_states,
        attentions=attentions,
        skew_threshold=0,
        entropy_threshold=1e5,
        core_positions=core_positions,
        icr_device=torch.device("cuda"),
    )

    icr_scores, _ = icr.compute_icr(
        top_k=5,
        top_p=None,
        pooling="mean",
        attention_uniform=False,
        hidden_uniform=False,
        use_induction_head=True,
    )

    # Average over generated tokens.
    # Result:
    # one number per transformer layer.
    features = [
        sum(layer_scores) / len(layer_scores)
        for layer_scores in icr_scores
    ]

    return generated_text, features


if __name__ == "__main__":

    test_prompt = "The capital of France is"

    answer, features = extract_icr_features(test_prompt)

    print("\nPrompt:")
    print(test_prompt)

    print("\nGenerated answer:")
    print(answer)

    print("\nICR feature vector:")
    print(features)

    print("\nNumber of features:")
    print(len(features))