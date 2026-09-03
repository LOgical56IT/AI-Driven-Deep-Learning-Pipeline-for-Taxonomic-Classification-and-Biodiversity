from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "server works"}from fastapi import FastAPI
