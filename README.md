# Two-Tower Recommender System

Рекомендательная система на основе архитектуры **Two-Tower (Dual Encoder)**, реализованная на датасете **MovieLens 100k**.

Цель проекта — исследовать современный подход к **candidate retrieval** и проверить его в сценарии **user cold start**, когда пользователи тестовой выборки не участвовали в обучении.

## Идея

Модель состоит из двух башен:

- **User Tower** — получает возраст, пол и профессию пользователя;
- **Item Tower** — получает `movie_id`, жанры и год выпуска фильма.

Обе башни преобразуют входные признаки в embedding-векторы. Релевантность фильма определяется через их скалярное произведение.

`user_id` не используется как обучаемый признак, поэтому модель может построить embedding для пользователя, отсутствующего в train.

> Текущая реализация решает user cold start при наличии профильных признаков, но не решает полноценный item cold start, поскольку Item Tower использует `movie_id`.

## Результаты

Для оценки используется user-level split:

- 755 пользователей — train;
- 188 пользователей — test;
- `K = 10`;
- релевантным считается фильм с рейтингом `>= 4`.

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|---|---:|---:|---:|---:|
| **Two-Tower** | 0.1043 | 0.0206 | 0.1091 | **0.0702** |
| **Popularity** | **0.3702** | **0.0812** | **0.3881** | 0.0059 |

Popularity показывает лучшую accuracy на этом датасете, а Two-Tower — значительно больший охват каталога: **7.02% против 0.59%**.

Таким образом, Two-Tower рассматривается прежде всего как **retrieval-компонент**, который в production может работать вместе с Popularity и отдельной ranking-моделью.

## Структура

```text
movie-recsys-two-tower/
├── data/
├── models/
├── src/
│   ├── data_loader.py
│   ├── two_tower.py
│   ├── train.py
│   └── evaluate.py
├── scripts/
│   └── run_all.py
├── reports/
│   └── metrics.csv
├── docs/
└── requirements.txt
```

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python -m src.train
python -m src.evaluate
```

Или полный запуск:

```bash
python scripts/run_all.py
```

Данные MovieLens должны находиться в `data/raw/`:

```text
u.data
u.item
u.user
```

## Технологии

- Python 3.11
- TensorFlow 2.15
- TensorFlow Recommenders 0.7.2
- pandas
- NumPy
- scikit-learn

Подробное описание архитектуры, эксперимента и production-аспектов находится в `docs/`.