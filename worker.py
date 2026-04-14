import asyncio
from datetime import datetime, timezone
from sqlalchemy import text,bindparam
from sqlalchemy.dialects.postgresql import JSONB

from db import AsyncSessionLocal
from face_utils import fetch_image_bytes, extract_face_descriptors

POLL_INTERVAL_SECONDS = 2

async def process_one_photo():
    async with AsyncSessionLocal() as db:

        # 🔹 pick one pending job (FIFO + lock)
        result = await db.execute(text("""
            SELECT id, url
            FROM photos
            WHERE indexing_status in ('pending','failed')
            ORDER BY id ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        """))

        row = result.fetchone()

        if not row:
            return False

        photo = dict(row._mapping)
        photo_id = photo["id"]

        print("Started the worker for photo_id:", photo_id)

        # 🔹 mark as processing
        await db.execute(text("""
            UPDATE photos
            SET indexing_status = 'processing',
                indexing_error = NULL
            WHERE id = :id
        """), {"id": photo_id})

        await db.commit()

        try:
            # 🔹 fetch + process
            image_bytes = await fetch_image_bytes(photo["url"])
            descriptors = extract_face_descriptors(image_bytes)

            # 🔹 success update
            await db.execute(text("""
                UPDATE photos
                SET face_descriptors = :descriptors,
                    faces_indexed = :faces_count,
                    indexing_status = 'done',
                    indexed_at = :indexed_at,
                    indexing_error = NULL
                WHERE id = :id
            """).bindparams(
                            bindparam("descriptors", type_=JSONB)
                        ), {
                "id": photo_id,
                "descriptors": descriptors,
                "faces_count": len(descriptors),
                "indexed_at": datetime.now(timezone.utc)
            })
            print("No of faces indexed for photo_id", photo_id, ":", len(descriptors))

            await db.commit()

        except Exception as e:
            print(e)
            # 🔹 failure update
            await db.execute(text("""
                UPDATE photos
                SET indexing_status = 'failed',
                    indexing_error = :error
                WHERE id = :id
            """), {
                "id": photo_id,
                "error": str(e)
            })

            await db.commit()

        return True


async def worker_loop():
    while True:
        found = await process_one_photo()
        if not found:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)