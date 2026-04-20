import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pipeline import AvatarExtractorPipeline

app = FastAPI(title="SAM-3D-Body Avatar Extractor")

pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    if pipeline is None:
        pipeline = AvatarExtractorPipeline()
        
def cleanup_temp_dir(dir_path: str):
    """Safely removes the temporary directory after the response is sent."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

@app.post("/extract-avatar")
async def extract_avatar(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported.")
        
    session_id = str(uuid.uuid4())
    temp_dir = os.path.join("temp", session_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process image
        mesh_path = pipeline.extract_avatar(file_path)
        
        if not os.path.exists(mesh_path):
            raise HTTPException(status_code=500, detail="Failed to generate 3D mesh.")
            
        # Register the cleanup task to run AFTER the file is sent
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
            
        return FileResponse(
            path=mesh_path,
            filename="avatar.obj",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        # If an error happens before returning, clean up immediately
        cleanup_temp_dir(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))
