import torch 
from src.icr_score import ICRScore

# Load the internal states we saved during generation.
data = torch.load("artifacts/qwen_internal_states.pt", map_location="cpu", with_weights=False)

hidden_states = data["hidden_states"]
attentions = data["attentions"]

# Number of tokens in the prompt.
input_len = hidden_states[0][0].shape[1]

print("Prompt:", data["prompt"])
print("Generated:", data["generated_text"])
print("Prompt length:", input_len)

# Tell ICR where the prompt ends and the generated tokens begin.
core_positions = {
    "user_prompt_start": 0,
    "user_prompt_end": input_len,
    "response_start": input_len,
}

# Move the stored tensors back to the GPU for ICR computation.
hidden_states = [tuple(x.cuda() for x in h) for h in hidden_states]
attentions = [tuple(x.cuda() for x in a) for a in attentions]

print("\nBuilding ICR object...")

icr = ICRScore(
    hidden_states=hidden_states,
    attentions=attentions,
    core_positions=core_positions,
    icr_device=torch.device("cuda"),
)

print ("ICR object created successfully.")