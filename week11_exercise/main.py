from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the output data model
class OutputData(BaseModel):
    prediction: str
    confidence: float

# Load the Iris dataset
iris = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Prepare the data
X = iris.drop("species", axis=1)
y = iris["species"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

# Save the trained pipeline
joblib.dump(pipeline, 'trained_pipeline.pkl')

@app.post("/predict", response_model=OutputData)
async def predict(input_data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make prediction
    prediction = pipeline.predict(input_df)
    
    # Get prediction probabilities
    proba = pipeline.predict_proba(input_df)
    
    # Get the confidence (probability) of the predicted class
    confidence = np.max(proba)
    
    return OutputData(prediction=prediction[0], confidence=float(confidence))

# Load the saved pipeline
loaded_pipeline = joblib.load('trained_pipeline.pkl')

@app.post("/predict_loaded", response_model=OutputData)
async def predict_loaded(input_data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make prediction using the loaded pipeline
    prediction = loaded_pipeline.predict(input_df)
    
    # Get prediction probabilities
    proba = loaded_pipeline.predict_proba(input_df)
    
    # Get the confidence (probability) of the predicted class
    confidence = np.max(proba)
    
    return OutputData(prediction=prediction[0], confidence=float(confidence))


# responsible for starting the web server that hosts your FastAPI application.
@app.get("/")
async def root():
    return {"message": "Welcome to the Iris classifier API"}

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)