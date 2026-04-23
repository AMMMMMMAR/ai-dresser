from fastapi import FastAPI
from tryon.router import router as tryon_router

app = FastAPI(
    title="VDS V2 API",
    description="Virtual Dressing System V2 — Body Measurement + Virtual Try-On Pipeline",
    version="1.0.0"
)


app.include_router(tryon_router)

@app.get("/health")
def health():
    return {"status": "ok"}