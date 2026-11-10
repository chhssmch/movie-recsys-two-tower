"""
Обучение Two-Tower в user-cold-start сценарии.

Ключевое исправление:
in-batch negatives предполагают один positive на query в батче.
Поэтому на каждой эпохе берём ровно один positive item на пользователя.
Иначе несколько строк одного user в одном batch превращают реальные positives
этого пользователя в false negatives.
"""

from pathlib import Path
import pandas as pd
import tensorflow as tf

from src.data_loader import (
    build_user_features,
    build_movie_features,
    build_interactions,
    build_all_interactions,
    split_users,
)
from src.two_tower import build_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def make_dataset(df: pd.DataFrame, batch_size: int = 256, seed: int = 42):
    feature_cols = [
        "user_id", "age", "gender", "occupation",
        "movie_id", "genres", "year",
    ]
    string_cols = {"user_id", "gender", "occupation", "movie_id", "genres"}

    features = {}
    for col in feature_cols:
        values = df[col].astype(str).to_numpy() if col in string_cols else df[col].to_numpy()
        features[col] = tf.constant(values)

    return (
        tf.data.Dataset.from_tensor_slices(features)
        .shuffle(len(df), seed=seed, reshuffle_each_iteration=False)
        .batch(batch_size)
    )


def one_positive_per_user(train_pos: pd.DataFrame, seed: int) -> pd.DataFrame:
    # Один positive на пользователя = нет duplicate queries внутри batch.
    return (
        train_pos.groupby("user_id", group_keys=False)
        .sample(n=1, random_state=seed)
        .reset_index(drop=True)
    )


def train_model(
    epochs: int = 30,
    embedding_dim: int = 64,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42,
):
    print("Загрузка данных...")
    users = build_user_features()
    movies = build_movie_features()
    positives = build_interactions(min_rating=4)
    all_interactions = build_all_interactions()

    train_users, test_users = split_users(
        all_interactions,
        test_fraction=0.2,
        seed=seed,
    )

    train_pos = (
        positives[positives["user_id"].isin(train_users)]
        .merge(users, on="user_id", how="inner")
        .merge(movies, on="movie_id", how="inner")
    )

    train_users_df = users[users["user_id"].isin(train_users)].reset_index(drop=True)

    print(f"Train users: {len(train_users)}")
    print(f"Test users (cold start): {len(test_users)}")
    print(f"Positive interactions in train: {len(train_pos)}")

    model = build_model(
        train_users_df,
        movies,
        embedding_dim=embedding_dim,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )

    for epoch in range(epochs):
        epoch_pos = one_positive_per_user(train_pos, seed + epoch)
        train_ds = make_dataset(
            epoch_pos,
            batch_size=batch_size,
            seed=seed + epoch,
        )

        history = model.fit(
            train_ds,
            epochs=1,
            verbose=1,
        )
        loss = float(history.history["loss"][-1])
        print(
            f"epoch {epoch + 1:02d}/{epochs} | "
            f"sampled users={len(epoch_pos)} | loss={loss:.4f}"
        )

    model_path = MODELS_DIR / "two_tower"
    model.save_weights(str(model_path) + ".weights.h5")
    print(f"Модель сохранена: {model_path}.weights.h5")

    return model, (users, movies, train_users, test_users, train_pos)


if __name__ == "__main__":
    train_model()
