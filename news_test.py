from datasets import load_dataset

dataset = load_dataset(
    "Brianferrell787/financial-news-multisource",
    "fnspid_news",
    streaming=True
)

train = dataset["train"]

first_row = next(iter(train))

print(first_row)
print(first_row.keys())