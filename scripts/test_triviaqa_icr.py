from datasets import load_dataset

from scripts.icr_pipeline import extract_icr_features


print("Loading 5 TriviaQA examples...")

dataset = load_dataset(
    "trivia_qa",
    "rc",
    split="validation[:5]"
)


for i, example in enumerate(dataset):

    question = example["question"]
    correct_answer = example["answer"]["value"]

    # Use one fixed prompt format for now.
    prompt = f"Question: {question}\nAnswer:"

    print(f"\n{'=' * 60}")
    print(f"Example {i + 1}")
    print("Question:", question)
    print("Correct answer:", correct_answer)

    generated_answer, features = extract_icr_features(
        prompt,
        max_new_tokens=10
    )

    print("Model answer:", generated_answer)
    print("ICR features:", features)