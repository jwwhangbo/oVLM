from fastapi import FastAPI

from ovlm.routers.api import router

app = FastAPI()
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Hello World"}
