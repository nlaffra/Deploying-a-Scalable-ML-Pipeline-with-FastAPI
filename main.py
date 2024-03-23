import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model
project_path = "/home/nlaffra/ml-pipeline-reset"
# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

encoder_path = os.path.join(project_path, "model", "encoder.pkl")
encoder = load_model(encoder_path)

model_path = os.path.join(project_path, "model", "model.pkl") 
model = load_model(model_path)

app = FastAPI(  title="Inference API",
                description="An API that takes a sample and runs an inference",
                version="1.0.0")

# load model artifacts on startup of the application to reduce latency
@app.on_event("startup")
async def startup_event(): 
    global model, encoder, lb
    # if saved model exits, load the model from disk
    if os.path.isfile(model_path):
        model = pickle.load(open(model_path, "rb"))
        encoder = pickle.load(open(encoder_path, "rb"))
        

@app.get("/")
async def greetings():
    return "Welcome to The Project API"


# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data_processed, _, _, _ = process_data(data, 
                                           categorical_features=cat_features, 
                                           training=False, 
                                           encoder=encoder, 
                                           lb=lb
                                          )
    _inference = model.predict(data_processed)
    return {"result": apply_label(_inference)}
