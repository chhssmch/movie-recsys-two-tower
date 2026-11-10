# Проблемы продакшена: Two-Tower

## 1. Общая архитектура

Учебный notebook и production RecSys — разные системы.

В production типичный pipeline выглядит так:

```text
events
  ↓
feature / interaction pipelines
  ↓
Two-Tower retrieval
  ↓
ANN index
  ↓
candidate set
  ↓
ranking model
  ↓
business rules
  ↓
recommendations
```

Поэтому deploy одной нейросети недостаточен.

## 2. In-batch negatives и sampling bias

### Проблема

In-batch negatives удобны, но:

- популярные items встречаются чаще;
- некоторые negatives на самом деле могут быть релевантными;
- распределение batch не обязательно совпадает с production traffic.

### Решения

- one-positive-per-query batching;
- popularity-aware sampling;
- hard negatives;
- sampling-bias correction;
- периодическая проверка распределения negatives.

### Недостаток

Сложная sampling strategy увеличивает стоимость подготовки данных и может ухудшить качество, если sampling distribution выбрана неправильно.

## 3. ANN search

Для 1682 items BruteForce подходит для учебного эксперимента.

Для большого каталога полный перебор становится узким местом.

Варианты:

- FAISS;
- HNSW;
- ScaNN.

Параметры ANN требуют trade-off между recall retrieval и latency.

### Недостаток

Приближённый поиск может не вернуть настоящий top-K. Поэтому ANN нужно оценивать отдельно: например, recall@K относительно exact BruteForce.

## 4. Embedding drift

Модель обучена на историческом поведении, но:

- появляются новые items;
- меняются вкусы пользователей;
- меняется каталог;
- меняются бизнес-приоритеты.

### Решения

- регулярное переобучение;
- versioning модели;
- versioning ANN index;
- пересчёт item embeddings;
- мониторинг распределений features/embeddings;
- rollback на предыдущую версию.

### Недостаток

Переобучение и пересборка индекса требуют вычислительных ресурсов и orchestration.

## 5. Cold start нового пользователя

### Проблема

У нового пользователя нет истории.

### Решение

Использовать profile/context features:

- возраст;
- устройство;
- время;
- язык;
- регион;
- первые действия пользователя.

В текущем учебном проекте используются age/gender/occupation.

### Ограничение

Profile features могут быть недостаточно информативными.

Поэтому production обычно использует hybrid strategy:

```text
new user
  → popularity / editorial / exploration
  → первые interactions
  → personalized retrieval
```

## 6. Cold start нового item

В текущей реализации item tower содержит movie_id embedding.

Поэтому настоящий item cold start отдельно не решён.

Для production можно добавить:

- категории;
- brand;
- текстовое описание;
- изображения;
- audio/video embeddings;
- metadata.

Тогда новый item можно обработать до появления достаточного количества interactions.

## 7. Feature pipeline

Качество retrieval зависит от качества features.

Проблемы:

- training-serving skew;
- пропуски;
- устаревшие значения;
- разные способы расчёта feature offline и online.

### Решения

- единый feature pipeline;
- feature store;
- schema validation;
- freshness monitoring;
- backfill и replay исторических features.

## 8. Latency и availability

Нужно контролировать:

- p50/p95/p99 latency;
- QPS;
- timeout rate;
- error rate;
- cache hit rate;
- ANN search latency.

Рекомендации должны иметь fallback:

```text
Two-Tower unavailable
       ↓
Popularity / trending / editorial
```

Система рекомендаций не должна ломать основной пользовательский сценарий.

## 9. Мониторинг ML

Офлайн:

- Precision@K;
- Recall@K;
- NDCG@K;
- catalog coverage;
- diversity;
- novelty.

Data quality:

- feature missing rate;
- distribution drift;
- embedding drift.

Online:

- retrieval recall;
- candidate acceptance rate;
- CTR;
- conversion;
- session length;
- retention.

## 10. A/B-тестирование

Офлайн-метрики не доказывают бизнес-эффект.

Для production:

```text
Control → current recommender
Test    → new recommender
```

Сравниваются:

- CTR;
- conversion;
- revenue / GMV;
- session length;
- retention.

Важно заранее определить primary metric и guardrail metrics.

## 11. Diversity и popularity bias

Two-Tower может выдавать много популярных items.

Поэтому кроме Precision нужно мониторить:

- catalog coverage;
- category coverage;
- diversity внутри top-K;
- долю head/mid/tail items.

Для бизнеса слишком высокая концентрация рекомендаций в head-каталоге может быть проблемой даже при хорошем CTR.

## 12. Итог

Production Two-Tower — это не только neural network.

Нужны:

1. feature pipeline;
2. training pipeline;
3. item embedding generation;
4. ANN index;
5. online user embedding;
6. fallback;
7. monitoring;
8. A/B testing;
9. регулярное обновление модели.

Учебная реализация проверяет только центральную часть этой архитектуры.
