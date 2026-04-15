"""
Face Swap Web App — FastAPI
Веб-интерфейс для замены лиц в GIF/MP4.
"""

import os
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from face_swapper import FaceSwapEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Face Swap App")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

engine = FaceSwapEngine()

# Прогресс задач: {task_id: {"status": ..., "progress": ..., "total": ..., "result": ...}}
tasks: Dict[str, dict] = {}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/swap")
async def swap_faces(
    media: UploadFile = File(...),
    face: UploadFile = File(...),
    face_only: str = "1",
):
    """Запускает замену лиц. Возвращает task_id для отслеживания прогресса."""
    # Валидация
    media_ext = Path(media.filename).suffix.lower()
    face_ext = Path(face.filename).suffix.lower()

    if media_ext not in (".gif", ".mp4", ".avi", ".mov", ".mkv", ".webm"):
        raise HTTPException(400, "Формат медиа не поддерживается. Нужен GIF или MP4.")
    if face_ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        raise HTTPException(400, "Формат фото не поддерживается. Нужен JPG/PNG.")

    task_id = str(uuid.uuid4())

    # Сохраняем файлы
    media_path = UPLOAD_DIR / f"{task_id}_media{media_ext}"
    face_path = UPLOAD_DIR / f"{task_id}_face{face_ext}"

    with open(media_path, "wb") as f:
        content = await media.read()
        f.write(content)

    with open(face_path, "wb") as f:
        content = await face.read()
        f.write(content)

    output_ext = media_ext
    output_path = OUTPUT_DIR / f"{task_id}_result{output_ext}"

    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "total": 0,
        "result": None,
        "error": None,
        "skipped": 0,       # кадров без лица
        "warnings": None,   # текст предупреждения для UI
    }

    # Запускаем обработку в фоне
    asyncio.get_event_loop().run_in_executor(
        None,
        _process_task,
        task_id,
        str(media_path),
        str(face_path),
        str(output_path),
        face_only != "0",
    )

    return {"task_id": task_id}


def _process_task(task_id: str, media_path: str, face_path: str, output_path: str, face_only: bool = True):
    """Фоновая обработка."""
    def progress_cb(current, total, skipped_frames=None):
        tasks[task_id]["progress"] = current
        tasks[task_id]["total"] = total
        if skipped_frames is not None:
            tasks[task_id]["skipped"] = len(skipped_frames)

    try:
        _, skipped_frames = engine.process(media_path, face_path, output_path, progress_callback=progress_cb, face_only=face_only)
        tasks[task_id]["status"] = "done"
        tasks[task_id]["result"] = os.path.basename(output_path)
        tasks[task_id]["skipped"] = len(skipped_frames)
        if skipped_frames:
            total = tasks[task_id]["total"] or 1
            tasks[task_id]["warnings"] = (
                f"На {len(skipped_frames)} кадрах из {total} лицо не обнаружено — эти кадры остались без замены."
            )
    except Exception as e:
        logger.exception("Ошибка обработки:")
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)
    finally:
        # Удаляем исходники
        try:
            os.unlink(media_path)
            os.unlink(face_path)
        except OSError:
            pass


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Возвращает прогресс задачи."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Задача не найдена")
    return task


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Скачивание результата."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(
        str(path),
        media_type="application/octet-stream",
        filename=filename,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
