
import pandas as pd
import utils.functions as functions
from transformers import AutoTokenizer
from utils.chunk_model import ChunkModel

CHUNK_SIZE = 156
NUM_TF_LAYERS = 2
HIDDEN_DIM = 768
EPOCHS = 3
DROPOUT_PROB = 0.2
TF_MODEL_NAME = "mediabiasgroup/magpie-babe-ft"

POOLING_STRATEGY = "mean"


print(f"MODEL: {TF_MODEL_NAME}")

print(
    f"CHUNK_SIZE {CHUNK_SIZE}, POOLING_STRATEGY {POOLING_STRATEGY}, NUM_TF_LAYERS {NUM_TF_LAYERS}, HIDDEN_DIM {HIDDEN_DIM}, EPOCHS {EPOCHS}, DROPOUT {DROPOUT_PROB}"
)

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

tokeniser = AutoTokenizer.from_pretrained(TF_MODEL_NAME)


tokenised_dataset = dataset.map(
    functions.tokenise_chunks,
    fn_kwargs={"tokeniser": tokeniser, "chunk_size": CHUNK_SIZE},
)

print(tokenised_dataset)


num_labels = len(pd.unique(train_df["labels"]))
train_labels = tokenised_dataset["train"]["labels"]
model = ChunkModel(
    tf_model_name=TF_MODEL_NAME,
    num_tf_layers=NUM_TF_LAYERS,
    hidden_dim=HIDDEN_DIM,
    num_classes=num_labels,
    train_labels=train_labels,
    pooling_strategy=POOLING_STRATEGY,
    dropout_prob=DROPOUT_PROB,
)
model = model.to(model.device)
print(model)

train_dataloader = model.batchify(tokenised_dataset["train"], batch_size=8)
valid_dataloader = model.batchify(tokenised_dataset["valid"], batch_size=8)

model.fit(train_dataloader, valid_dataloader, epochs=EPOCHS)

model.predict(tokenised_dataset["test"])
