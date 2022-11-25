import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import Counter, defaultdict

from sklearn import preprocessing

import pickle

raw_df = pd.read_csv("data/train_set.csv")

        #'trip_id', # input
        #'duration',
        #'start_time',
        #'end_time',
        #'start_lat',
        #'start_lon',
        #'end_lat',
        #'end_lon',
        #'bike_id',
        #'plan_duration',
        #'trip_route_category',
        #'passholder_type', # Target
        #'start_station',
        #'end_station'

# normalize duration

raw_df["duration"] = raw_df["duration"]/raw_df["duration"].max()

# La duración se ve bien

# Bike id tiene basura, se borra el texto y se quita "a" de los
# pocos que lo tienen

def clean_nums(x):
    try:
        return int(x)
    except:
        if x.endswith("a"):
            return int(x[:-1])
        return np.nan

raw_df["bike_id"] = raw_df["bike_id"].apply(clean_nums)

le_bike_id = preprocessing.LabelEncoder()
raw_df["bike_id"] = le_bike_id.fit_transform(raw_df["bike_id"])
# or one hot encode

# Los días de plan duration pasan los 365 ... se llevan a nan

raw_df.loc[raw_df["plan_duration"] > 365, "plan_duration"] = np.nan

# Trip category move to 1 and -1

le_trip = preprocessing.LabelEncoder()
raw_df["trip_route_category"] = le_trip.fit_transform(raw_df["trip_route_category"])
raw_df["trip_route_category"] = (raw_df["trip_route_category"] - 0.5) * 2

# passholder_type is target, change to int label

raw_df["passholder_type"] = raw_df["passholder_type"].fillna("nan")
le_passholder = preprocessing.LabelEncoder()
raw_df["passholder_type"] = le_passholder.fit_transform(raw_df["passholder_type"])

#lb = preprocessing.LabelBinarizer()
#ohe_pass = lb.fit_transform(raw_df["passholder_type"])
#ohe_pass = pd.DataFrame(ohe_pass,
#        columns=["passholder_type_%d"%i for i in range(len(ohe_pass[1]))])
#raw_df = pd.concat([raw_df, ohe_pass], axis=1)

# Valores de tiempo

raw_df["start_time"] = raw_df["start_time"].apply(pd.to_datetime)
raw_df["end_time"] = raw_df["end_time"].apply(pd.to_datetime)

base = "start_time"
raw_df["{}_{}".format(base, "year")] = raw_df[base].dt.year
raw_df["{}_{}".format(base, "month")] = raw_df[base].dt.month
raw_df["{}_{}".format(base, "day")] = raw_df[base].dt.day
raw_df["{}_{}".format(base, "hour")] = raw_df[base].dt.hour
raw_df["{}_{}".format(base, "minute")] = raw_df[base].dt.minute

base = "end_time"
raw_df["{}_{}".format(base, "year")] = raw_df[base].dt.year
raw_df["{}_{}".format(base, "month")] = raw_df[base].dt.month
raw_df["{}_{}".format(base, "day")] = raw_df[base].dt.day
raw_df["{}_{}".format(base, "hour")] = raw_df[base].dt.hour
raw_df["{}_{}".format(base, "minute")] = raw_df[base].dt.minute


# Varios datos de latitud y longitud estan errados
# se sustituyen por Nan

# Hay una instancia que falta un -

raw_df.loc[raw_df.start_lon > 100, "start_lon"] *= -1

# El resto se llevan a Nan

raw_df.loc[raw_df.start_lon > 0, ['start_lat', "start_lon"]] = np.nan
raw_df.loc[raw_df.end_lon > 0, ['end_lat', "end_lon"]] = np.nan

# Latitud y longitud son redundantes con la location id
# Se va a normalizar sobre cada campo 

s_stat_dict = defaultdict(list)
e_stat_dict = defaultdict(list)
for (lid, row) in tqdm(raw_df.iterrows(), total=len(raw_df)):
    s_loc = row[['start_lat', "start_lon"]].to_numpy().astype("float32")
    e_loc = row[['end_lat', "end_lon"]].to_numpy().astype("float32")
    if not np.isnan(s_loc).any():
        s_stat_dict[row["start_station"]].append(tuple(s_loc))
    if not np.isnan(e_loc).any():
        e_stat_dict[row["end_station"]].append(tuple(e_loc))

s_stat_to_l = dict()
for k, v in s_stat_dict.items():
    c = Counter(v)
    s_stat_to_l[k] = c.most_common(1)[0][0]
e_stat_to_l = dict()
for k, v in e_stat_dict.items():
    c = Counter(v)
    e_stat_to_l[k] = c.most_common(1)[0][0]

for (_, row) in tqdm(raw_df.iterrows(), total=len(raw_df)):
    s_stat = row.start_station
    e_stat = row.end_station
    if s_stat in s_stat_to_l:
        row.loc[["start_lat", "start_lon"]] = s_stat_to_l[s_stat]
    if e_stat in e_stat_to_l:
        row.loc[["end_lat", "end_lon"]] = e_stat_to_l[e_stat]

le_start_s = preprocessing.LabelEncoder()
raw_df["start_station"] = le_start_s.fit_transform(raw_df["start_station"])
le_end_s = preprocessing.LabelEncoder()
raw_df["end_station"] = le_end_s.fit_transform(raw_df["end_station"])

raw_df = raw_df.drop("start_time", axis=1)
raw_df = raw_df.drop("end_time", axis=1)


raw_df = raw_df.replace(np.nan, -1)
raw_df.to_csv("data/cool_data.csv", index=False)

with open("data/le_bike_id.pkl", "wb") as f:
    pickle.dump(le_bike_id, f)
with open("data/le_trip.pkl", "wb") as f:
    pickle.dump(le_trip, f)
with open("data/le_passholder.pkl", "wb") as f:
    pickle.dump(le_passholder, f)
with open("data/le_start_s.pkl", "wb") as f:
    pickle.dump(le_start_s, f)
with open("data/le_end_s.pkl", "wb") as f:
    pickle.dump(le_end_s, f)
