# Face Swap App

Веб-приложение для замены лиц в GIF и MP4. Загружаешь гифку/видео + фото лица — получаешь результат с заменённым лицом. Выражение и ориентация лица на каждом кадре сохраняются от оригинала.

## Стек

- **Бэкенд:** Python 3.10 + FastAPI + Uvicorn
- **Face Swap:** InsightFace (buffalo_l детектор + inswapper_128 модель замены)
- **Видео:** OpenCV + imageio + ffmpeg
- **Фронт:** Чистый HTML/CSS/JS (без фреймворков)

## Быстрый старт

### 1. Установка зависимостей

```bash
# Клонируем / распаковываем проект
cd face-swap-app

# Виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Зависимости
pip install -r requirements.txt
```

### 2. Скачивание моделей

**A) Автоматически** (buffalo_l скачается сам при первом запуске):

```bash
bash download_models.sh
```

**B) Вручную** — нужен файл `inswapper_128.onnx` (~554 MB):

> ⚠️ Оригинальный репо `deepinsight/inswapper` на HuggingFace требует авторизацию (401) — не использовать.

**Вариант 1 — GitHub FaceFusion (рекомендуется):**
```bash
mkdir -p models
wget -O models/inswapper_128.onnx \
  "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
```

**Вариант 2 — HuggingFace (публичный репо, без авторизации, проверено — работает):**
```bash
mkdir -p models
wget -O models/inswapper_128.onnx \
  "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
```

**Вариант 3 — Google Drive:**
```bash
pip install gdown
gdown "1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF" -O models/inswapper_128.onnx
```

### 3. Запуск

```bash
python app.py
```

Открыть в браузере: **http://localhost:8000**

---

## Docker

```bash
# Сначала скачай модель в models/
mkdir -p models
wget -O models/inswapper_128.onnx \
  "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"

# Сборка и запуск
docker compose up -d --build

# Открыть: http://localhost:8000
```

### С GPU (NVIDIA)

В `docker-compose.yml` раскомментируй секцию `deploy.resources` и убедись, что установлен `nvidia-container-toolkit`.

Также в `requirements.txt` замени:
```
onnxruntime==1.19.2
```
на:
```
onnxruntime-gpu==1.19.2
```

---

## Как работает

1. **Загрузка** — пользователь загружает GIF/MP4 + фото лица через веб-интерфейс
2. **Детекция** — InsightFace (buffalo_l) находит лицо на фото-источнике и извлекает эмбеддинг
3. **Покадровая обработка** — на каждом кадре GIF/видео:
   - Находим все лица
   - Модель inswapper_128 заменяет каждое найденное лицо на лицо из фото
   - Замена сохраняет положение, размер, ориентацию и мимику от оригинального кадра
4. **Сборка** — кадры собираются обратно в GIF/MP4 (с аудио через ffmpeg)
5. **Результат** — превью + скачивание

## Структура проекта

```
face-swap-app/
├── app.py                 # FastAPI сервер
├── face_swapper.py        # Движок замены лиц
├── requirements.txt       # Python зависимости
├── Dockerfile
├── docker-compose.yml
├── download_models.sh     # Скрипт загрузки моделей
├── templates/
│   └── index.html         # Веб-интерфейс
├── models/                # Сюда кладутся модели (не в git)
├── uploads/               # Временные загрузки
└── outputs/               # Результаты
```

## Системные требования

- **CPU:** Работает, но медленно (1-3 сек/кадр)
- **GPU (NVIDIA):** Рекомендуется для видео (0.1-0.3 сек/кадр)
- **RAM:** Минимум 4 GB, рекомендуется 8 GB
- **ffmpeg:** Нужен для обработки аудио в видео
