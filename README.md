# Исследование адаптивной разкадровки (Google Colab)

Подбор FPS нарезки видео для photogrammetry: нарезка кадров (OpenCV) + пробный sparse SfM (**PyCOLMAP**).

## Структура

```text
adaptive_frame_sampling/
├── colab.ipynb
├── requirements.txt
├── adaptive_sampling/
│   ├── common/              # пути, загрузка YAML
│   ├── frame_extraction/    # метод 1: нарезка кадров
│   ├── sparse_eval/         # метод 2: sparse PyCOLMAP
│   └── ml/                  # метод 3: ML пары good/bad
├── configs/
├── data/
└── results/
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

---

## Метод 3 — ML (хорошая / плохая пара кадров)

Модель: **ResNet18** (два кадра → общий энкодер → бинарный класс).

### Порядок работы

**1. Sparse-прогоны** (уже сделано) → `results/task2_sparse_eval/`

**2. Шаблон Excel для разметки:**

```bash
python -m adaptive_sampling.ml.build_dataset --export-template
```

Создаётся `data/labels/manual/pairs.xlsx` с листом `pairs`:
- соседние пары кадров;
- `suggested_label` — подсказка из sparse (`good`/`bad`);
- **`label`** — заполняете вручную: `good` или `bad` (можно `хорошая`/`плохая`).

Подробнее: `data/labels/README.md`

**3. Разметка в Excel**

Откройте `pairs.xlsx`, для каждой строки укажите `label`. Можно добавлять свои строки (новые видео / пары). Ориентир — `comparison_table.csv` из sparse.

**4. Сборка датасета и обучение (Colab, GPU):**

```python
!pip install -q -r requirements.txt

import sys
sys.path.insert(0, ".")

# после заполнения pairs.xlsx:
!python -m adaptive_sampling.ml.build_dataset
!python -m adaptive_sampling.ml.train
```

Чекпоинт: `models/checkpoints/best.pt`

**5. Инференс на новых кадрах:**

```python
!python -m adaptive_sampling.ml.predict \
  --frames-dir data/frames/video_2026-04-16_11-31-49/fps_5 \
  --checkpoint models/checkpoints/best.pt \
  --output results/ml/pair_predictions.csv
```

### Минимум данных

- **≥ 50–100** размеченных пар для первого эксперимента;
- лучше размечать пары с **разных FPS** и видео (в т.ч. провальные sparse-прогоны);
- обязательно помечайте **bad** на scene cut, даже если весь FPS в среднем «хороший».
