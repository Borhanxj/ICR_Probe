import os # For saving the internals to disk.
import torch # For tensor operations and model inference.
from transformers import AutoTokenizer, AutoModelForCausalLM # For loading the model and tokenizer.

model_name = "Qwen/Qwen3-0.6B" # The model we want to use. You can change this to any other model that supports causal language modeling.

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name) # Load the tokenizer for the specified model. This will be used to convert text into token IDs and vice versa.

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained( # Load the model for causal language modeling. This model will be used to generate text based on the input prompt.
    model_name,
    dtype=torch.float16,
    attn_implementation="eager",
).to("cuda")

model.eval() # Set the model to evaluation mode. This is important because it disables certain layers like dropout that are only used during training.

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# These will store:
# index 0 -> internals for the whole prompt
# index 1 -> internals for generated token 1
# index 2 -> internals for generated token 2
# ...
hidden_states = [] # A list that stores the hidden states of the model at each step of generation. Hidden states are the outputs of each layer in the model and can be used to understand how the model processes the input. The size of it will be the same as the number of layers in the model, and each element will have a shape corresponding to (batch_size, sequence_length, hidden_size).
attentions = [] # A list that stores the attention weights of the model at each step of generation. Attention weights indicate how much focus the model gives to different parts of the input when generating each token. The size of it will naturally be the same as hidden_states.

generated_tokens = [] # A list that stores the token IDs of the generated tokens. This will be used to reconstruct the generated text after the generation process is complete.

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
    next_token = torch.argmax( # Select the token with the highest probability from the model's output logits for the last position in the sequence. This will be the next token to generate.
        outputs.logits[:, -1, :],
        dim=-1
    ).unsqueeze(0)

    # Generate at most 10 tokens.
    for step in range(10):

        generated_tokens.append(next_token.item())

        # Feed only the new token.
        # The cache remembers all previous tokens.
        outputs = model( # This is where the model generates the next token based on the previous tokens. The model takes the last generated token and the cached past key values (which contain information about all previously generated tokens) to produce the next token in the sequence.
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
        next_token = torch.argmax( # Select the token with the highest probability from the model's output logits for the last position in the sequence. This will be the next token to generate.
            outputs.logits[:, -1, :],
            dim=-1
        ).unsqueeze(0)

        if next_token.item() == tokenizer.eos_token_id: # If the model predicts the end-of-sequence token, we stop generating further tokens. This is a common stopping criterion in text generation tasks to prevent generating unnecessary or irrelevant tokens after the intended output has been completed.
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