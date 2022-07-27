import fastapi
import joblib
import uvicorn
from fastapi import FastAPI
import numpy as np
import joblib
from pydantic.main import BaseModel

app = FastAPI(title = "My ML App")

@app.get("/")
def home():
    return "API is working as expected."


class MyData(BaseModel):
    X : int

@app.post("/predict")
def predict(data:MyData):
    my_model = joblib.load("app/my_model")
    X = np.array(data.X).reshape(-1,1)
    return str(my_model.predict(X))

if __name__ == "__main__" :
    uvicorn.run(app, host="127.0.0.1", port=8000)