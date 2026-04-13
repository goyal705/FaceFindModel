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

def preprocess_for_occlusion(img):
    """Enhance image for better detection of occluded faces"""
    # Increase brightness/contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE on L channel — improves contrast locally
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

def extract_face_descriptors(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return []

    h, w = img.shape[:2]
    if w > 640:
        scale = 640 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    app = get_face_app()
    
    faces = app.get(img)

    if not faces:
        enhanced = preprocess_for_occlusion(img)
        faces = app.get(enhanced)

    if not faces:
        brightened = cv2.convertScaleAbs(img, alpha=1.3, beta=20)
        faces = app.get(brightened)

    if not faces:
        return []

    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )[:5]

    descriptors = []
    for face in faces:
        if getattr(face, "det_score", 0) < 0.5:  # lowered from 0.8
            continue

        emb = face.embedding
        if emb is None:
            continue

        norm = np.linalg.norm(emb)
        if norm != 0:
            emb = emb / norm

        descriptors.append(emb.tolist())

    return descriptors