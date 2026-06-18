# Исследование адаптивной разкадровки (Google Colab)

Подбор FPS нарезки видео для photogrammetry: нарезка кадров (OpenCV) + пробный sparse SfM (**PyCOLMAP**).

## Структура

```text
adaptive_frame_sampling/
├── colab.ipynb              # главный ноутбук
├── requirements.txt
├── adaptive_sampling/       # Python-код
├── configs/
│   ├── extraction_fps.yaml  # FPS: 2, 5, 10, 30, 60
│   └── sparse_eval.yaml     # PyCOLMAP + пороги метрик
├── data/
│   ├── videos/              # исходные видео
│   └── frames/                # нарезанные кадры
└── results/                   # метрики и сводки
```

## Быстрый старт в Colab

1. Загрузите репозиторий / папку `adaptive_frame_sampling` в Colab (или клонируйте).
2. Откройте `colab.ipynb`.
3. Положите видео в `data/videos/`.
4. Запустите все ячейки.

Установка зависимостей (первая ячейка ноутбука):

```python
!pip install -q -r requirements.txt
```

Программный запуск:

```python
import sys
sys.path.insert(0, ".")  # корень adaptive_frame_sampling

from adaptive_sampling import (
    extract_all_fps_in_directory,
    run_batch_for_video,
)

extract_all_fps_in_directory("data/videos")

results = run_batch_for_video("my_video_slug")
```

## Результаты sparse eval

Для каждого FPS:

- `results/sparse_eval/<video>/fps_<N>/sparse_metrics.json`
- `results/sparse_eval/<video>/fps_<N>/sparse_metrics.xlsx`

После batch:

- `results/sparse_eval/_batch/<video>/comparison_table.xlsx`
- `results/sparse_eval/_batch/<video>/top_fps_modes.json`

## GPU в Colab

В `configs/sparse_eval.yaml`:

```yaml
pycolmap:
  device: cuda
  matcher: sequential   # не exhaustive!
  max_images: 120       # лимит кадров на прогон
```

**Почему долго:** `match_exhaustive` сравнивает все пары кадров — при fps_30/60 на минутном ролике это тысячи изображений и миллионы пар. Для видео всегда используйте `sequential` и `max_images`.

Ориентир на Colab T4: один FPS-режим ~2–8 мин при `max_images: 120`; все 5 FPS одного видео ~15–40 мин.

## Связь с BLK-06

Исследование автономно. По итогам пороги и рекомендуемый FPS переносятся в production вручную.
