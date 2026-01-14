from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health():
    return {"status": "running"}

@app.get("/predict")
def predict(x: float):
    return {
        "input": x,
        "prediction": x * 2,
        "confidence": 0.95
    }
