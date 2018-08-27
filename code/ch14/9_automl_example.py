import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


y_label_df = pd.read_csv("./data/train_label.csv")
y = pd.Series([1,1,0,1], index=["2month", "month", "retained","week"])
y_label_df["class"] = y_label_df["label"].map(y)


X_raw_df = pd.read_csv("./data/train_activity.csv")
X_df = X_raw_df.groupby(["acc_id"]).sum().reset_index()

from sklearn.model_selection import train_test_split

X_payment_df = pd.read_csv("./data/train_payment.csv")
X_df = X_df.merge(X_payment_df.groupby("acc_id").sum().reset_index(), how="left", on="acc_id")


X_df = X_df.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X_df.values, y_label_df["class"].values, test_size=0.2, stratify=y_label_df["class"])

rfc = RandomForestClassifier()
rfc.fit(X_train[:, 1:], y_train)
from sklearn.metrics import accuracy_score
y_pred = rfc.predict(X_test[:,1:])
print(accuracy_score(y_test, y_pred))

import autosklearn.classification
automl = autosklearn.classification.AutoSklearnClassifier(
    )

automl.fit(X_train[:, 1:], y_train)

y_pred = automl.predict(X_test[:,1:])
print("Accuracy score", accuracy_score(y_test, y_pred))
