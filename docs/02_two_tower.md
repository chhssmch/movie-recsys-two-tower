# Two-Tower: как работает модель

## 1. Общая идея

Two-Tower (dual encoder) — архитектура для retrieval в рекомендательных системах.

В проекте:

```text
User features ──► User Tower ──► user_emb (64)
                                      │
                                      │ dot product
                                      ▼
Item features ──► Item Tower ──► item_emb (64)
```

Скалярное произведение определяет score:

`score(user, item) = dot(user_emb, item_emb)`

Item embeddings можно заранее посчитать для всего каталога. В production вместо полного перебора используется ANN-индекс.

## 2. User Tower

### Входные признаки

В текущей версии:

- `age` → Normalization;
- `gender` → one-hot;
- `occupation` → Embedding.

`user_id` передаётся в dataset только для совместимости, но **не используется как обучаемый признак**.

Это принципиальное изменение для cold-start эксперимента: тестовые пользователи отсутствуют в train, поэтому их ID нельзя использовать как источник персонализации.

Структура:

```text
age ────────────────► Normalization ──┐
gender ─────────────► one-hot ────────┼─► concat
occupation ─────────► Embedding ──────┘
                                      │
                                Dense(128, ReLU)
                                      │
                                Dropout(0.1)
                                      │
                                  Dense(64)
                                      │
                                  user_emb
```

## 3. Item Tower

Вход:

- `movie_id` → Embedding;
- `genres` → multi-hot;
- `year` → Normalization.

Структура:

```text
movie_id ─────────► Embedding(64) ──┐
genres ───────────► multi-hot ──────┼─► concat
year ─────────────► Normalization ──┘
                                    │
                              Dense(128, ReLU)
                                    │
                              Dropout(0.1)
                                    │
                                Dense(64)
                                    │
                                item_emb
```

### Ограничение

В этой работе item cold start отдельно не тестируется.

Поскольку item tower содержит `movie_id` embedding, нельзя утверждать, что новый фильм без обученного ID embedding полностью решается этой моделью. Для настоящего item cold start нужно сделать item tower feature-only или добавить новые metadata/text/audio features.

## 4. Обучение

Используется retrieval loss с in-batch negatives.

В батче:

```text
(user_1, item_1)
(user_2, item_2)
...
(user_N, item_N)
```

Для `user_i` его `item_i` — positive, а остальные items используются как negatives.

### Важная проблема исходной реализации

В исходном коде один и тот же пользователь мог встречаться несколько раз в одном batch, потому что один пользователь имеет много позитивных фильмов.

Тогда реальный positive:

```text
(user_1, movie_A)
(user_1, movie_B)
```

может для строки `movie_A` попасть в diagonal-negative позицию как `movie_B`.

Это false negative.

### Исправление

На каждой эпохе из train interactions выбирается **ровно один positive на пользователя**.

Так batch содержит уникальных пользователей и не создаёт false negatives из-за повторяющихся query users.

Это не устраняет все возможные false negatives: один и тот же фильм может нравиться разным пользователям. Но существенно уменьшает систематическую проблему исходной реализации.

## 5. Почему модель может работать для нового пользователя

Тестовый пользователь не был представлен в обучении.

Но его:

- возраст,
- пол,
- профессия

доступны системе.

Поэтому User Tower может построить embedding без истории взаимодействий.

Это и есть проверяемый в проекте user cold start.

## 6. Inference

Офлайн:

```text
все фильмы
   ↓
Item Tower
   ↓
item embeddings
   ↓
ANN index
```

Онлайн:

```text
profile нового пользователя
   ↓
User Tower
   ↓
user embedding
   ↓
nearest-neighbor search
   ↓
Top-K candidates
```

В учебной реализации используется BruteForce, потому что каталог MovieLens 100k содержит всего 1682 фильма.

## 7. Ограничения

1. Нет последовательного контекста.
2. Profile features могут быть слишком грубыми.
3. In-batch negatives неидеальны.
4. MovieLens 100k мал для демонстрации реального преимущества deep retrieval.
5. Модель является retrieval-моделью, а не полноценным ranking stack.
6. Интерпретируемость ниже, чем у простых эвристик.

## 8. Параметры исправленной реализации

- embedding dimension: 64;
- hidden layer: 128;
- dropout: 0.1;
- optimizer: Adam;
- learning rate: 0.001;
- epochs: 30;
- batch size: 256;
- positive: rating >= 4;
- split: 80% train users / 20% unseen test users.

Параметры являются стартовой конфигурацией. Итоговое качество нужно измерять после повторного запуска.

## 9. Главный вывод

Two-Tower в этом проекте следует позиционировать не как «модель, которая гарантированно побеждает baseline», а как современный retrieval-механизм, проверенный в отдельном user-cold-start эксперименте.
