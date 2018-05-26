import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def transform_status(x):
    if "Mrs" in x or "Ms" in x:
        return "Mrs"
    elif "Mr" in x:
        return "Mr"
    elif "Miss" in x:
        return "Miss"
    elif "Master" in x:
        return "Master"
    elif "Dr" in x:
        return "Dr"
    elif "Rev" in x:
        return "Rev"
    elif "Col" in x:
        return "Col"
    else:
        return "0"

train_df = pd.read_csv("titanic/train.csv")
test_df = pd.read_csv("titanic/test.csv")

train_id = train_df["PassengerId"].values
test_id = test_df["PassengerId"].values

all_df = train_df.append(test_df).set_index('PassengerId')
all_df["Sex"] = all_df["Sex"].replace({"male":0,"female":1})
all_df["Age"].fillna(
    all_df.groupby("Pclass")["Age"].transform("mean"), inplace=True)
all_df["cabin_count"] = all_df["Cabin"].map(lambda x : len(x.split()) if type(x) == str else 0)
all_df["social_status"] = all_df["Name"].map(lambda x : transform_status(x))
all_df = all_df.drop([62,830])
train_id = np.delete(train_id, [62-1,830-1])
all_df.loc[all_df["Fare"].isnull(), "Fare"] = 12.415462
all_df["cabin_type"] = all_df["Cabin"].map(lambda x : x[0] if type(x) == str else "99")

del all_df["Cabin"]
del all_df["Name"]
del all_df["Ticket"]

y = all_df.loc[train_id, "Survived"].values
del all_df["Survived"]

X_df = pd.get_dummies(all_df)
X = X_df.values

minmax_scaler = MinMaxScaler()
minmax_scaler.fit(X)
X = minmax_scaler.transform(X)

X_train = X[:len(train_id)]
X_test = X[len(train_id):]

np.save("tatanic_X_train.npy", X_train)
np.save("tatanic_y_train.npy", y)
np.save("tatanic_test.npy", X_test)
