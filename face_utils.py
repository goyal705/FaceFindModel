import cv2
import httpx
import numpy as np
import json

from app.core.face_engine import get_face_app

async def fetch_image_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.get(url)
        res.raise_for_status()
        return res.content

def extract_face_descriptors(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return []

    h, w = img.shape[:2]
    if w > 640:
        scale = 640 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    faces = get_face_app().get(img)

    if not faces:
        return []

    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )[:3]
    faces = faces[:3]
    descriptors = []
    for face in faces:
        if getattr(face, "det_score", 0) < 0.8:
            continue

        emb = face.embedding
        norm = np.linalg.norm(emb)
        if norm != 0:
            emb = emb / norm

        descriptors.append(emb.tolist())

    return json.dumps(descriptors)