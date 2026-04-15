"""
Face Swap Engine — InsightFace + inswapper_128
Обрабатывает GIF/MP4: находит лицо на каждом кадре, заменяет его лицом из фото,
сохраняя положение, размер и ориентацию исходного лица.
"""

import os
import cv2
import numpy as np
import imageio
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Optional

import insightface
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


class FaceSwapEngine:
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or MODELS_DIR
        self._app = None
        self._swapper = None

    def _ensure_loaded(self):
        """Ленивая инициализация моделей."""
        if self._app is not None:
            return

        logger.info("Загрузка моделей InsightFace...")

        # Детектор + распознаватель лиц
        self._app = FaceAnalysis(
            name="buffalo_l",
            root=self.model_dir,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))

        # Модель замены лиц
        swapper_path = os.path.join(self.model_dir, "inswapper_128.onnx")
        if not os.path.exists(swapper_path):
            # Пробуем найти в стандартных путях insightface
            home_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
            if os.path.exists(home_path):
                swapper_path = home_path
            else:
                raise FileNotFoundError(
                    f"Модель inswapper_128.onnx не найдена.\n"
                    f"Скачайте её и положите в: {swapper_path}\n"
                    f"Или в: {home_path}"
                )

        self._swapper = insightface.model_zoo.get_model(
            swapper_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        logger.info("Модели загружены.")

    def get_source_face(self, image_path: str):
        """Извлекает лицо из фото-источника."""
        self._ensure_loaded()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")

        faces = self._app.get(img)
        if not faces:
            raise ValueError("Лицо не найдено на загруженном фото.")

        # Берём самое крупное лицо
        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        return faces[0]

    @staticmethod
    def _build_face_only_mask(frame_shape: tuple, landmarks: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """
        Строит маску, покрывающую только область лица (без волос:
        от лба до подбородка, от уха до уха).
        landmarks — 106-точечные (есть), иначе используем kps (5 точек).
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        if landmarks is not None and len(landmarks) >= 68:
            # 68-точечные ландмарки: контур лица 0-16,
            # брови 17-26, нос 27-35, глаза 36-47, рот 48-67
            # Контур лица (0-16) + верхняя часть бровей (17-26) в обратном порядке
            chin = landmarks[0:17].astype(np.int32)
            brow_top = landmarks[17:27].astype(np.int32)
            # Полигон лица: контур + верх бровей в обратном направлении
            hull_pts = np.concatenate([chin, brow_top[::-1]])
        elif landmarks is not None and len(landmarks) >= 5:
            hull_pts = landmarks[:5].astype(np.int32)
        elif kps is not None:
            hull_pts = kps.astype(np.int32)
        else:
            # Фоллбэк: маска всего кадра
            mask[:] = 1.0
            return mask

        cv2.fillConvexPoly(mask, cv2.convexHull(hull_pts), 1.0)

        # Размываем границу маски для плавного перехода
        blur_r = max(int(min(h, w) * 0.02) | 1, 11)
        mask = cv2.GaussianBlur(mask, (blur_r, blur_r), 0)
        return mask

    def swap_frame(self, frame: np.ndarray, source_face, face_only: bool = True) -> tuple:
        """Заменяет все лица в кадре на source_face.
        face_only=True: вставляет только область лица, волосы остаются из оригинального кадра.
        Возвращает (result_frame, faces_found: bool).
        """
        self._ensure_loaded()
        faces = self._app.get(frame)
        if not faces:
            return frame, False

        result = frame.copy()
        for face in faces:
            swapped = self._swapper.get(result, face, source_face, paste_back=True)

            if face_only:
                # Определяем ландмарки: InsightFace может давать
                # face.landmark_2d_106, face.landmark_3d_68 или face.kps
                lmk = getattr(face, "landmark_2d_106", None)
                if lmk is None:
                    lmk = getattr(face, "landmark_3d_68", None)
                    if lmk is not None:
                        lmk = lmk[:, :2]  # берём только XY
                kps = getattr(face, "kps", None)

                mask = self._build_face_only_mask(frame.shape, lmk, kps)
                # mask: float32 [0..1], shape (H, W)
                # Блендинг: result = swapped * mask + original * (1 - mask)
                mask3 = mask[:, :, np.newaxis]  # (H, W, 1)
                result = (swapped.astype(np.float32) * mask3
                          + result.astype(np.float32) * (1.0 - mask3)).astype(np.uint8)
            else:
                result = swapped

        return result, True

    def process_gif(
        self,
        gif_path: str,
        face_image_path: str,
        output_path: str,
        progress_callback=None,
        face_only: bool = True,
    ) -> str:
        """Обрабатывает GIF: заменяет лица покадрово."""
        source_face = self.get_source_face(face_image_path)

        reader = imageio.get_reader(gif_path)
        meta = reader.get_meta_data()
        duration = meta.get("duration", 100)  # мс на кадр
        fps = 1000.0 / duration if duration > 0 else 10

        frames = list(reader)
        total = len(frames)
        logger.info(f"GIF: {total} кадров, {fps:.1f} FPS")

        swapped_frames = []
        skipped_frames = []
        for i, frame in enumerate(frames):
            # imageio отдаёт RGB, конвертируем в BGR для OpenCV
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            swapped, found = self.swap_frame(bgr, source_face, face_only=face_only)
            if not found:
                skipped_frames.append(i + 1)
                logger.warning(f"Кадр {i + 1}/{total}: лицо не найдено, кадр пропущен без замены")
            # Обратно в RGB
            rgb = cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)
            swapped_frames.append(rgb)

            if progress_callback:
                progress_callback(i + 1, total, skipped_frames)

        # Сохраняем GIF
        imageio.mimsave(output_path, swapped_frames, duration=duration / 1000.0, loop=0)
        if skipped_frames:
            logger.warning(f"GIF сохранён с {len(skipped_frames)} кадрами без замены: {output_path}")
        else:
            logger.info(f"GIF сохранён: {output_path}")
        return output_path, skipped_frames

    def process_video(
        self,
        video_path: str,
        face_image_path: str,
        output_path: str,
        progress_callback=None,
        face_only: bool = True,
    ) -> str:
        """Обрабатывает MP4: заменяет лица покадрово."""
        source_face = self.get_source_face(face_image_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Видео: {total} кадров, {fps:.1f} FPS, {width}x{height}")

        # Пишем во временный файл без звука
        tmp_video = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_video, fourcc, fps, (width, height))

        frame_idx = 0
        skipped_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            swapped, found = self.swap_frame(frame, source_face, face_only=face_only)
            if not found:
                skipped_frames.append(frame_idx)
                logger.warning(f"Кадр {frame_idx}/{total}: лицо не найдено, кадр пропущен без замены")
            writer.write(swapped)

            if progress_callback:
                progress_callback(frame_idx, total, skipped_frames)

        cap.release()
        writer.release()

        # Копируем аудио из оригинала через ffmpeg
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", tmp_video,
                    "-i", video_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-shortest",
                    output_path,
                ],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # ffmpeg не установлен или нет аудио — просто копируем
            import shutil
            shutil.move(tmp_video, output_path)
        else:
            os.unlink(tmp_video)

        if skipped_frames:
            logger.warning(f"Видео сохранено с {len(skipped_frames)} кадрами без замены: {output_path}")
        else:
            logger.info(f"Видео сохранено: {output_path}")
        return output_path, skipped_frames

    def process(
        self,
        media_path: str,
        face_image_path: str,
        output_path: str,
        progress_callback=None,
        face_only: bool = True,
    ) -> tuple:
        """Автодетект формата и обработка.
        Возвращает (output_path, skipped_frames: list[int]).
        """
        ext = Path(media_path).suffix.lower()
        if ext == ".gif":
            return self.process_gif(media_path, face_image_path, output_path, progress_callback, face_only=face_only)
        elif ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            return self.process_video(media_path, face_image_path, output_path, progress_callback, face_only=face_only)
        else:
            raise ValueError(f"Неподдерживаемый формат: {ext}")
