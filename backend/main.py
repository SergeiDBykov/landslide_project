from fastapi import FastAPI
from backend.model.predict import predict

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/predict")
async def get_prediction(lat: float = 0, lon: float = 10, month: int = 0):
    risk, fig = predict(lon, lat, month)
    return {"message": f"Success", "risk": risk, "fig": fig}

