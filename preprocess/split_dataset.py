import sys

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
CLASS_RANGES = [(0, 24), (24.01, 32.00), (32.01, 40.00), (40.01, 64)]

CLASSES = ["Problematic", "Questionable", "Generally Reliable", "Reliable"]

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"dataset {DATASET_VERSION}")

df = pd.read_csv(
    f"../dataset/scraped_clean_{DATASET_VERSION}.csv",
    index_col=0,
)

outlets_df = pd.read_csv(
    "../original_dataset/BAT/ad_fontes/outlets_classes_scores.csv",
    index_col=0,
    encoding="ISO-8859-1",
)


def map_to_class(score):
    for i, (start, end) in enumerate(CLASS_RANGES):
        if start <= score <= end:
            return i, CLASSES[i]


df[["labels", "class"]] = df["reliability_score"].apply(map_to_class).apply(pd.Series)
df.to_csv(f"../dataset/scraped_clean_{DATASET_VERSION}.csv")


outlets_df["outlet_labels"] = outlets_df["reliability_score"].apply(
    lambda x: map_to_class(x)[0]
)

outlets_df["adfontes_url"] = outlets_df["url"]

df.drop(columns="outlet", inplace=True)
df = df.merge(
    outlets_df[["adfontes_url", "outlet_labels", "outlet"]], on=["adfontes_url"]
)

grouped_df = df.groupby(["labels", "outlet"])
# grouped_df = df.groupby(["labels"])

train_data, test_data, valid_data = [], [], []

for group_name, group_df in grouped_df:
    if len(group_df) <= 5:
        train_data.append(group_df)
    else:
        train_group, test_group = train_test_split(
            group_df, test_size=0.22, random_state=SEED
        )
        test_group, valid_group = train_test_split(
            test_group, test_size=0.4, random_state=SEED
        )

        train_data.append(train_group)
        test_data.append(test_group)
        valid_data.append(valid_group)


train_set = pd.concat(train_data)
test_set = pd.concat(test_data)
valid_set = pd.concat(valid_data)


train_set = train_set.sample(frac=1, random_state=SEED).reset_index(drop=True)
test_set = test_set.sample(frac=1, random_state=SEED).reset_index(drop=True)
valid_set = valid_set.sample(frac=1, random_state=SEED).reset_index(drop=True)

train_set.to_csv(f"../dataset/{DATASET_VERSION}/train.csv")
test_set.to_csv(f"../dataset/{DATASET_VERSION}/test.csv")
valid_set.to_csv(f"../dataset/{DATASET_VERSION}/valid.csv")

# outlets_df.to_csv("../dataset/outlets.csv")

print("TRAIN")
print(train_set.shape)
print(train_set["labels"].value_counts())
# print(train_set["outlet"].value_counts(normalize=True))

print("TEST")
print(test_set.shape)
print(test_set["labels"].value_counts())
# print(test_set["outlet"].value_counts(normalize=True))

print("VALID")
print(valid_set.shape)
print(valid_set["labels"].value_counts())
# print(valid_set["outlet"].value_counts(normalize=True))
