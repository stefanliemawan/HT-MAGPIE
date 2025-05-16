import os
import platform
import sys

import pandas as pd
import torch
import utils.functions as functions
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "allenai/longformer-base-4096"

print(f"MODEL: {MODEL_NAME}")

train_df = pd.read_csv(f"../dataset/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/valid.csv", index_col=0)

# train_df, test_df, valid_df = functions.generate_title_content_features(
#     train_df, test_df, valid_df
# )
train_df, test_df, valid_df = functions.generate_outlet_title_content_features(
    train_df, test_df, valid_df
)

dataset = functions.create_dataset(train_df, test_df, valid_df)
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenised_dataset = dataset.map(
    lambda x: tokeniser(
        x["features"], padding="max_length", truncation=True, max_length=4096
    ),
    batched=True,
)

print(tokenised_dataset)

functions.print_class_distribution(tokenised_dataset)

num_labels = len(pd.unique(train_df["labels"]))
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, max_length=4096
)
if platform.system() == "Darwin":
    model = model.to("mps")
elif torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")

functions.train(tokenised_dataset, model, epochs=4)
