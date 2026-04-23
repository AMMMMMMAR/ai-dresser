from fastapi import FastAPI
from measurement.router import router as measurement_router

app = FastAPI(
    title="VDS V2 API",
    description="Virtual Dressing System V2 — Body Measurement Pipeline",
    version="1.0.0"
)

app.include_router(measurement_router)

@app.get("/health")
def health():
    return {"status": "ok"}