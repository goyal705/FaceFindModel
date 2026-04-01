import asyncio
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from db import get_db
from worker import worker_loop
from app.core.face_engine import init_face_app
from face_utils import extract_face_descriptors
app = FastAPI(title="Face Index Service")

@app.on_event("startup")
async def startup_event():
    print("Starting the application")
    init_face_app()
    asyncio.create_task(worker_loop())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/index-photo/{photo_id}")
async def trigger_index(photo_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
            text(f"SELECT * FROM photos WHERE id = :photo_id"),
            {"photo_id": photo_id}
        )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Photo not found")

    photo = dict(row._mapping)

    if photo["indexing_status"] == "done":
        return {"status": "already_done", "photo_id": photo_id}

    if photo["indexing_status"] == "processing":
        return {"status": "already_processing", "photo_id": photo_id}

    await db.execute(
        text("""
            UPDATE photos
            SET indexing_status = 'pending',
                indexing_error = NULL
            WHERE id = :photo_id
        """),
        {"photo_id": photo_id}
    )

    await db.commit()

    return {"status": "queued", "photo_id": photo_id}

@app.post("/audience/extract-face")
async def extract_user_face(file: UploadFile = File(...)):
    print("Received a request to extract face descriptors from uploaded file:", file.filename)
    contents = await file.read()

    try:
        user_face = extract_face_descriptors(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face extraction failed: {str(e)}")

    if not user_face:
        return {
            "status": "no_face",
            "descriptors": []
        }

    return {
        "status": "success",
        "descriptors": user_face
    }