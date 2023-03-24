from datasets import load_dataset

dataset = load_dataset("nielsr/funsd", split="train")

print(dataset[0])
