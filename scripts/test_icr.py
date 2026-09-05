import torch 
from src.icr_score import ICRScore

# Load the internal states we saved during generation.
data = torch.load("artifacts/qwen_test_internals.pt", map_location="cpu", weights_only=False)

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
hidden_states = [tuple(x.cuda().float()  for x in h) for h in hidden_states]
attentions = [tuple(x.cuda().float()  for x in a) for a in attentions]

print("\nBuilding ICR object...")

icr = ICRScore(
    hidden_states=hidden_states,
    attentions=attentions,
    skew_threshold=0,
    entropy_threshold=1e5,
    core_positions=core_positions,
    icr_device=torch.device("cuda"),
)

print ("ICR object created successfully.")

print("\nComputing ICR score...")
icr_scores, top_p_mean = icr.compute_icr(
    top_k=5,
    top_p=None,
    pooling="mean",
    attention_uniform=False,
    hidden_uniform=False,
    use_induction_head=True,
)

print("\nICR computation completed.")

print("\nNumber of layers:", len(icr_scores))
print("Number of generated tokens:", len(icr_scores[0]))

print("\nICR scores for layer 1:")
print(icr_scores[0])

print("\nMean selected attention proportion:")
print(top_p_mean)

layer_averages = [
    sum(layer_scores) / len(layer_scores)
    for layer_scores in icr_scores
]

print("\nProbe feature vector:")
print(layer_averages)

print("\nFeature count:")
print(len(layer_averages))