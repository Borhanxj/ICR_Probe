from datasets import load_dataset

print("Loading TriviaQA...")

dataset = load_dataset(
    "trivia_qa",
    "rc",
    split="validation[:5]"
)

for i, example in enumerate(dataset):
    print(f"\n--- Example {i + 1} ---")
    print("Question:", example["question"])
    print("Main answer:", example["answer"]["value"])
    print("Accepted aliases:", example["answer"]["aliases"])