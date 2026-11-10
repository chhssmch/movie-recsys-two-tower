"""
Оценка Two-Tower в user-cold-start сценарии.

Split:
- train users: используются для обучения;
- test users: полностью unseen во время обучения;
- relevant: rating >= 4 у test users;
- Popularity строится ТОЛЬКО по train users (иначе leakage).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from src.data_loader import (
    build_user_features,
    build_movie_features,
    build_interactions,
    build_all_interactions,
    split_users,
)
from src.two_tower import build_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def precision_at_k(recs, relevant, k):
    return sum(m in relevant for m in recs[:k]) / k


def recall_at_k(recs, relevant, k):
    return (
        sum(m in relevant for m in recs[:k]) / len(relevant)
        if relevant else 0.0
    )


def ndcg_at_k(recs, relevant, k):
    hits = [
        1.0 / np.log2(i + 2)
        for i, movie_id in enumerate(recs[:k])
        if movie_id in relevant
    ]
    dcg = sum(hits)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def build_index(model, movies_df, batch_size=256):
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_tower)

    movie_ds = tf.data.Dataset.from_tensor_slices({
        "movie_id": tf.constant(movies_df["movie_id"].astype(str).values),
        "genres": tf.constant(movies_df["genres"].astype(str).values),
        "year": tf.constant(movies_df["year"].values, dtype=tf.int32),
    }).batch(batch_size)

    index.index_from_dataset(
        movie_ds.map(
            lambda f: (f["movie_id"], model.item_tower(f))
        )
    )
    return index


def two_tower_recommend(index, user_row, k=10, exclude=None, catalog_size=None):
    exclude = exclude or set()
    candidate_k = min(
        k + len(exclude),
        catalog_size or (k + len(exclude))
    )

    query = {
        "age": tf.constant([user_row["age"]], dtype=tf.float32),
        "gender": tf.constant([str(user_row["gender"])]),
        "occupation": tf.constant([str(user_row["occupation"])]),
    }

    _, titles = index(query, k=candidate_k)
    raw = titles[0].numpy()

    recs = [
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in raw
    ]

    return [m for m in recs if m not in exclude][:k]

def build_popularity(train_pos):
    return (
        train_pos.groupby("movie_id")
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )


def evaluate_model(
    recommend_fn,
    users_df,
    test_df,
    k=10,
    max_users=None,
):
    user_lookup = users_df.set_index("user_id").to_dict("index")

    precisions, recalls, ndcgs = [], [], []
    all_recs = set()

    test_user_ids = test_df["user_id"].drop_duplicates().tolist()
    if max_users is not None:
        test_user_ids = test_user_ids[:max_users]

    for user_id in test_user_ids:
        relevant = set(
            test_df.loc[
                (test_df["user_id"] == user_id)
                & (test_df["rating"] >= 4),
                "movie_id",
            ]
        )
        if not relevant or user_id not in user_lookup:
            continue

        recs = recommend_fn(user_lookup[user_id])

        all_recs.update(recs)
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

    return precisions, recalls, ndcgs, all_recs


def summarize(precisions, recalls, ndcgs, all_recs, catalog_size, n_users):
    return {
        "Precision@10": round(float(np.mean(precisions)), 4),
        "Recall@10": round(float(np.mean(recalls)), 4),
        "NDCG@10": round(float(np.mean(ndcgs)), 4),
        "Coverage": round(len(all_recs) / catalog_size, 4),
        "n_users": n_users,
    }


if __name__ == "__main__":
    users = build_user_features()
    movies = build_movie_features()
    positives = build_interactions(min_rating=4)
    all_interactions = build_all_interactions()

    train_users, test_users = split_users(
        all_interactions,
        test_fraction=0.2,
        seed=42,
    )

    train_pos = positives[positives["user_id"].isin(train_users)].reset_index(drop=True)
    test_all = all_interactions[all_interactions["user_id"].isin(test_users)].reset_index(drop=True)

    train_users_df = users[users["user_id"].isin(train_users)].reset_index(drop=True)

    print(f"Train users: {len(train_users)}")
    print(f"Test users: {len(test_users)}")
    print(f"Train positives: {len(train_pos)}")
    print(f"Test interactions: {len(test_all)}")

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

    def tt_recommend(user_row):
        return two_tower_recommend(index, user_row, k=10, catalog_size=len(movies))

    tt = evaluate_model(
        tt_recommend,
        users,
        test_all,
        k=10,
    )
    tt_metrics = summarize(
        *tt,
        catalog_size=len(movies),
        n_users=len(tt[0]),
    )

    popular_ids = build_popularity(train_pos)

    def popularity_recommend(user_row):
        return popular_ids[:10]

    pop = evaluate_model(
        popularity_recommend,
        users,
        test_all,
        k=10,
    )
    pop_metrics = summarize(
        *pop,
        catalog_size=len(movies),
        n_users=len(pop[0]),
    )

    df = pd.DataFrame([
        {"model": "Two-Tower", **tt_metrics},
        {"model": "Popularity", **pop_metrics},
    ]).set_index("model")

    print("\nИТОГОВАЯ ТАБЛИЦА")
    print(df.to_string())

    out = Path(__file__).resolve().parent.parent / "reports" / "metrics.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out)
    print(f"\nСохранено: {out}")
