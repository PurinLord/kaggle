import pandas as pd
import numpy as np
from sklearn import ensemble
import pickle


data = pd.read_csv("data/cool_data.csv")
data = data.replace(np.nan, -1)

x = data.drop(["passholder_type", "plan_duration"], axis=1)
y = data["passholder_type"]

rf = ensemble.RandomForestClassifier()

print("Start training")

rf.fit(x, y)


test = pd.read_csv("data/cool_test_data.csv")

t_pred = rf.predict(test)


with open("data/le_passholder.pkl", "rb") as f:
    le_passholder = pickle.load(f)
labs = le_passholder.inverse_transform(t_pred)

sub = test.join(pd.DataFrame(labs,
    columns=["passholder_type"]))[["trip_id", "passholder_type"]]

sub.to_csv("data/sub.csv", index=False)
