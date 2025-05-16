import numpy as np
import pandas as pd
import tensorflow as tf
import utils.functions as functions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

train_df = pd.read_csv(f"../dataset/train.csv", index_col=0)
test_df = pd.read_csv(f"../dataset/test.csv", index_col=0)
valid_df = pd.read_csv(f"../dataset/valid.csv", index_col=0)

train_df, test_df, valid_df = functions.generate_title_content_features(
    train_df, test_df, valid_df
)

vectorizer = CountVectorizer()

x_train = vectorizer.fit_transform(train_df["features"].values)
x_test = vectorizer.transform(test_df["features"].values)
x_valid = vectorizer.transform(valid_df["features"].values)

x_train = x_train.toarray()
x_valid = x_valid.toarray()
x_test = x_test.toarray()

num_labels = len(pd.unique(train_df["labels"]))

y_train = train_df["labels"].values
y_test = test_df["labels"].values
y_valid = valid_df["labels"].values


print(x_train.shape)
print(x_test.shape)


class_weights = np.asarray(
    compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
).astype(np.float32)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(class_weights_dict)

y_train = to_categorical(y_train, num_classes=num_labels)
y_valid = to_categorical(y_valid, num_classes=num_labels)

model = Sequential()
model.add(
    Dense(
        128,
        input_dim=x_train.shape[1],
        activation="relu",
    )
)
model.add(Dropout(0.2))
model.add(Dense(num_labels, activation="softmax"))

optimiser = tf.keras.optimizers.AdamW(learning_rate=2e-5)

model.compile(
    loss="categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=8,
    validation_data=(x_valid, y_valid),
    class_weight=class_weights_dict,
    # callbacks=[lr_scheduler],
)

predictions = model.predict(x_test)
y_pred = predictions.argmax(axis=1)
print(classification_report(y_test, y_pred))


precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)

print(
    {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
)
