from fastapi import FastAPI
from backend.model.predict import predict

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/test")
async def hello_world():
    return {"message": f"It works!"}

