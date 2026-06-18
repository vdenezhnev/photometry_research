# Разметка пар кадров для ML

## Файл разметки

`manual/pairs.xlsx` — ручная разметка. Создаётся командой:

```bash
python -m adaptive_sampling.ml.build_dataset --export-template
```

## Лист `pairs`

| Столбец | Обязательно | Описание |
|---------|-------------|----------|
| `video_slug` | да | Имя видео без расширения |
| `fps_label` | да | `fps_2`, `fps_5`, … |
| `frame_a` | да | Имя файла, напр. `frame_000001.jpg` |
| `frame_b` | да | Следующий кадр, напр. `frame_000002.jpg` |
| `suggested_label` | нет | Подсказка из sparse (`good` / `bad`), не трогать |
| `label` | **для обучения** | Ваша метка: `good` или `bad` |
| `notes` | нет | Комментарий |

### Как размечать

- **good** — пара подходит для SfM: достаточное перекрытие, нет резкой смены сцены.
- **bad** — низкое перекрытие, размытие, scene cut, провал на sparse для этого FPS.

Смотрите сводку sparse: `results/task2_sparse_eval/_batch/<video>/comparison_table.csv`.

Если для `fps_5` регистрация 82%, а для `fps_2` — 55%, пары на `fps_2` чаще **bad**, на `fps_5` — **good**.

### Добавление новых видео

1. Положите видео в `data/videos/`, нарежьте кадры.
2. Запустите sparse-прогон.
3. Снова `--export-template` (допишет новые строки) или скопируйте строки вручную.
4. Заполните столбец `label`.

## Обучение

Только строки с заполненным `label`. Минимум ~50–100 размеченных пар.

```bash
python -m adaptive_sampling.ml.build_dataset
python -m adaptive_sampling.ml.train
```
