from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

import pandas as pd

class Info(BaseModel):
    ps_id: int

app = FastAPI()
df = pd.read_csv("data/sub_1.csv", index_col=0)

def predict(ps_id):
    return str(df.loc[ps_id]["passholder_type"])

@app.exception_handler(RequestValidationError)
def unicorn_exception_handler():
    return {"message": "Error."}

@app.get("/{ps_id}")
def root(ps_id):
    try:
        passholder_type = predict(int(ps_id))
        return {"passholder_type": passholder_type}
    except:
        return {"Error": "Not Found"}
