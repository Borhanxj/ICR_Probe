import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    attn_implementation="eager",
).to("cuda")

model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# These will store:
# index 0 -> internals for the whole prompt
# index 1 -> internals for generated token 1
# index 2 -> internals for generated token 2
# ...
hidden_states = []
attentions = []

generated_tokens = []

with torch.no_grad():

    # First, process the whole prompt.
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

    # Choose the first generated token.
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1
    ).unsqueeze(0)

    # Generate at most 10 tokens.
    for step in range(10):

        generated_tokens.append(next_token.item())

        # Feed only the new token.
        # The cache remembers all previous tokens.
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

        # Predict the following token.
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

print("\nPrompt:")
print(prompt)

print("\nGenerated:")
print(generated_text)

print("\nStored generation steps:", len(hidden_states))

print("\nPrompt hidden-state shape:")
print(hidden_states[0][0].shape)

print("\nFirst generated token hidden-state shape:")
print(hidden_states[1][0].shape)

print("\nPrompt attention shape:")
print(attentions[0][0].shape)

print("\nFirst generated token attention shape:")
print(attentions[1][0].shape)


# Save everything so we can give it to ICR later.
os.makedirs("artifacts", exist_ok=True)

torch.save(
    {
        "prompt": prompt,
        "generated_text": generated_text,
        "hidden_states": hidden_states,
        "attentions": attentions,
    },
    "artifacts/qwen_test_internals.pt",
)

print("\nSaved to artifacts/qwen_test_internals.pt")