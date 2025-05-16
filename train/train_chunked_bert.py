import os
import platform

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import functions
from utils.chunk_model import ChunkModel

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME = "bert-base-cased"

CHUNK_SIZE = 512
EPOCHS = 4

print(
    f"WINDOW_SIZE: {CHUNK_SIZE}"
)
print(f"MODEL: {MODEL_NAME}")

train_df = pd.read_csv(f"../dataset/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/valid.csv", index_col=0)


train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)
# train_df, test_df, valid_df = functions.generate_outlet_title_content_features(
#     train_df, test_df, valid_df
# )

dataset = functions.create_dataset(train_df, test_df, valid_df)

tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenised_dataset = dataset.map(
    functions.tokenise_chunks,
    fn_kwargs={
        "tokeniser": tokeniser,
        "chunk_size": CHUNK_SIZE,
    },
)

print(tokenised_dataset)


class Model(ChunkModel):
    def __init__(
        self,
        tf_model_name,
        hidden_dim,
        num_classes,
        train_labels,
        dropout_prob=0,
    ):
        super(ChunkModel, self).__init__()
        if platform.system() == "Darwin":
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tf_model_name = tf_model_name
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes

        self.init_layers()
        self.calculate_class_weights(train_labels)
        self.init_loss_optimiser()

    def init_layers(self):
        self.tf_model = AutoModelForSequenceClassification.from_pretrained(
            self.tf_model_name, num_labels=self.num_classes
        )
        self.tf_model = self.tf_model.to(self.device)

    def forward(self, input_ids, attention_mask):
        tf_model_output = self.tf_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        logits = tf_model_output.logits

        return logits


num_labels = len(pd.unique(train_df["labels"]))
train_labels = tokenised_dataset["train"]["labels"]
model = Model(
    tf_model_name=MODEL_NAME,
    hidden_dim=None,
    num_classes=num_labels,
    train_labels=train_labels,
    dropout_prob=0,
)
model = model.to(model.device)
print(model)

train_dataloader = model.batchify(tokenised_dataset["train"], batch_size=8)
valid_dataloader = model.batchify(tokenised_dataset["valid"], batch_size=8)

model.fit(train_dataloader, valid_dataloader, epochs=EPOCHS)

model.predict(tokenised_dataset["test"])
