#!/bin/bash
# Скачивание моделей InsightFace для Face Swap App
# Запускать один раз перед первым стартом.

set -e

MODELS_DIR="$(dirname "$0")/models"
mkdir -p "$MODELS_DIR"

echo "=== Скачивание buffalo_l (детектор лиц) ==="
# InsightFace скачает автоматически при первом запуске,
# но можно ускорить, скачав заранее:
python3 -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', root='$MODELS_DIR')
app.prepare(ctx_id=0, det_size=(640, 640))
print('buffalo_l — OK')
"

echo ""
echo "=== Модель inswapper_128.onnx ==="

SWAPPER="$MODELS_DIR/inswapper_128.onnx"
if [ -f "$SWAPPER" ]; then
    echo "inswapper_128.onnx уже есть."
else
    echo "ВНИМАНИЕ: Файл inswapper_128.onnx нужно скачать вручную."
    echo ""
    echo "Варианты:"
    echo "1) Hugging Face:  https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
    echo "2) Google Drive (оригинал):  https://drive.google.com/file/d/1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu"
    echo ""
    echo "Скачайте и положите файл сюда:"
    echo "  $SWAPPER"
    echo ""
    echo "Или в: ~/.insightface/models/inswapper_128.onnx"
fi

echo ""
echo "=== Готово ==="
