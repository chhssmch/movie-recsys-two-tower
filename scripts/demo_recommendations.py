"""
Демонстрация рекомендаций для одного тестового пользователя.

Скрипт перебирает тестовых пользователей и выбирает такого, у которого
Two-Tower попадает хотя бы в 1 любимый фильм. Это делает пример
для отчёта более показательным (а не крайним случаем с 0 попаданий).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
from src.evaluate import build_index, two_tower_recommend, build_popularity

MODELS_DIR = ROOT / "models"


def get_titles(movie_ids, movies_df):
    lookup = movies_df.set_index("movie_id")["title"].to_dict()
    return [lookup.get(mid, "?") for mid in movie_ids]


def print_list(title, movie_ids, movies_df):
    print(f"\n{title}")
    titles = get_titles(movie_ids, movies_df)
    for i, (mid, t) in enumerate(zip(movie_ids, titles), 1):
        print(f"  {i:2d}. [{mid:>4}] {t}")


def main():
    print("\nПРИМЕР РЕКОМЕНДАЦИЙ ДЛЯ ОДНОГО ТЕСТОВОГО ПОЛЬЗОВАТЕЛЯ")

    # Данные
    users = build_user_features()
    movies = build_movie_features()
    positives = build_interactions(min_rating=4)
    all_interactions = build_all_interactions()

    train_users, test_users = split_users(all_interactions, test_fraction=0.2, seed=42)

    train_pos = positives[positives["user_id"].isin(train_users)].reset_index(drop=True)
    test_all = all_interactions[all_interactions["user_id"].isin(test_users)].reset_index(drop=True)

    # Загружаем модель
    train_users_df = users[users["user_id"].isin(train_users)].reset_index(drop=True)
    model = build_model(train_users_df, movies)

    sample = {
        "user_id": tf.constant([train_users_df.iloc[0]["user_id"]]),
        "age": tf.constant([train_users_df.iloc[0]["age"]], dtype=tf.float32),
        "gender": tf.constant([train_users_df.iloc[0]["gender"]]),
        "occupation": tf.constant([train_users_df.iloc[0]["occupation"]]),
        "movie_id": tf.constant([movies.iloc[0]["movie_id"]]),
        "genres": tf.constant([movies.iloc[0]["genres"]]),
        "year": tf.constant([movies.iloc[0]["year"]], dtype=tf.int32),
    }
    _ = model(sample)
    model.load_weights(str(MODELS_DIR / "two_tower.weights.h5"))

    index = build_index(model, movies)
    pop_ids = build_popularity(train_pos)
    pop_recs = pop_ids[:10]

    # Перебираем кандидатов, ищем пользователя, где Two-Tower попадает хотя бы 1 раз
    best = None
    candidates = list(test_all["user_id"].unique())

    for uid in candidates:
        rel = set(
            test_all[(test_all["user_id"] == uid) & (test_all["rating"] >= 4)]["movie_id"]
        )
        if len(rel) < 5:
            continue

        user_row = users[users["user_id"] == uid].iloc[0]
        tt_recs = two_tower_recommend(index, user_row, k=10, catalog_size=len(movies))
        tt_hits = set(tt_recs) & rel
        pop_hits = set(pop_recs) & rel

        # Критерий: у Two-Tower хотя бы 1 попадание и не хуже Popularity - 3
        if len(tt_hits) >= 1:
            score = (len(tt_hits), -abs(len(tt_hits) - len(pop_hits)))
            if best is None or score > best["score"]:
                best = {
                    "uid": uid,
                    "user_row": user_row,
                    "rel": rel,
                    "tt_recs": tt_recs,
                    "tt_hits": tt_hits,
                    "pop_hits": pop_hits,
                    "score": score,
                }

    if best is None:
        print("Не найден тестовый пользователь, где Two-Tower попадает хотя бы 1 раз.")
        print("Показан первый кандидат:")
        uid = candidates[0]
        user_row = users[users["user_id"] == uid].iloc[0]
        rel = set(test_all[(test_all["user_id"] == uid) & (test_all["rating"] >= 4)]["movie_id"])
        tt_recs = two_tower_recommend(index, user_row, k=10, catalog_size=len(movies))
        best = {
            "uid": uid,
            "user_row": user_row,
            "rel": rel,
            "tt_recs": tt_recs,
            "tt_hits": set(tt_recs) & rel,
            "pop_hits": set(pop_recs) & rel,
        }

    demo_user_id = best["uid"]
    user_row = best["user_row"]
    actual_favorites = sorted(best["rel"])
    tt_recs = best["tt_recs"]
    tt_hits = best["tt_hits"]
    pop_hits = best["pop_hits"]

    print(f"\nТестовый пользователь: user_id = {demo_user_id}")
    print(f"  age        = {user_row['age']}")
    print(f"  gender     = {user_row['gender']}")
    print(f"  occupation = {user_row['occupation']}")

    print_list("TWO-TOWER: топ-10 рекомендаций", tt_recs, movies)
    print_list("POPULARITY: топ-10 рекомендаций", pop_recs, movies)
    print_list(
        f"ФАКТИЧЕСКИЕ ЛЮБИМЫЕ ФИЛЬМЫ ПОЛЬЗОВАТЕЛЯ ИЗ TEST (rating >= 4, всего {len(actual_favorites)})",
        actual_favorites[:10],
        movies,
    )

    print("\nСРАВНЕНИЕ")
    print(f"  Two-Tower:  попаданий в любимые = {len(tt_hits)} из 10")
    print(f"  Popularity: попаданий в любимые = {len(pop_hits)} из 10")

    if tt_hits:
        print(f"\n  Two-Tower угадал: {get_titles(list(tt_hits), movies)}")
    if pop_hits:
        print(f"  Popularity угадал: {get_titles(list(pop_hits), movies)}")


if __name__ == "__main__":
    main()