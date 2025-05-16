import sys

import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

try:
    DATASET_VERSION = sys.argv[1]
except IndexError:
    DATASET_VERSION = "vx"

print(f"dataset {DATASET_VERSION}")

train_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/{DATASET_VERSION}/valid.csv", index_col=0)

# do majority votes based on outlet and record the f1

outlet_majority_label = (
    train_df.groupby("outlet")["labels"].apply(lambda x: x.mode()[0]).reset_index()
)

# print(outlet_majority_label)


def predict_majority_label(outlet):
    if outlet in outlet_majority_label["outlet"].values:
        return outlet_majority_label.loc[
            outlet_majority_label["outlet"] == outlet, "labels"
        ].values[0]
    else:
        return None  # Handle case when outlet is not found in the DataFrame


def calculate_metrics(df):
    preds = df["outlet"].apply(predict_majority_label).values
    labels = df["labels"]

    report = classification_report(labels, preds, zero_division=1)
    print(f"\n{report}")

    precision = precision_score(labels, preds, average="weighted", zero_division=1)
    recall = recall_score(labels, preds, average="weighted", zero_division=1)
    f1 = f1_score(labels, preds, average="weighted", zero_division=1)

    print(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


# print("TRAIN")
# calculate_metrics(train_df)
print("TEST")
calculate_metrics(test_df)
# print("VALID")
# calculate_metrics(valid_df)

# TEST


#               precision    recall  f1-score   support

#            0       0.56      0.70      0.62        27
#            1       0.58      0.46      0.52        54
#            2       0.56      0.53      0.54       104
#            3       0.91      0.93      0.92       384

#     accuracy                           0.80       569
#    macro avg       0.65      0.66      0.65       569
# weighted avg       0.79      0.80      0.80       569

# {'precision': 0.7945671180199992, 'recall': 0.7996485061511424, 'f1': 0.7959329549331157}
