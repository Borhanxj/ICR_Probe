from datasets import load_dataset

print("Loading TriviaQA...")

dataset = load_dataset(
    "mandarjoshi/trivia_qa",
    "rc.nocontext",
    split="validation"
)

for i, example in enumerate(dataset):
    print(f"\n--- Example {i + 1} ---")
    print("Question:", example["question"])
    print("Main answer:", example["answer"]["value"])
    print("Accepted aliases:", example["answer"]["aliases"])