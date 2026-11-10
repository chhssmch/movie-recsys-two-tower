"""Единый запуск: train → evaluate."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import train_model
from src.evaluate import (
    evaluate_model, summarize, build_index,
    two_tower_recommend, build_popularity,
)
from src.data_loader import (
    build_user_features, build_movie_features,
    build_interactions, build_all_interactions, split_users,
)
import pandas as pd
import tensorflow as tf
from src.two_tower import build_model

MODELS_DIR = ROOT / "models"


def main():
    t0 = time.time()
    print("TWO-TOWER RECSYS — ПОЛНЫЙ ПАЙПЛАЙН")

    model, (users, movies, train_users, test_users, train_pos) = train_model()

    # Оценка
    all_interactions = build_all_interactions()
    test_all = all_interactions[all_interactions["user_id"].isin(test_users)].reset_index(drop=True)

    index = build_index(model, movies)

    def tt_recommend(user_row):
        return two_tower_recommend(index, user_row, k=10, catalog_size=len(movies))

    tt = evaluate_model(tt_recommend, users, test_all, k=10)
    tt_metrics = summarize(*tt, catalog_size=len(movies), n_users=len(tt[0]))

    popular_ids = build_popularity(train_pos)

    def pop_recommend(user_row):
        return popular_ids[:10]

    pop = evaluate_model(pop_recommend, users, test_all, k=10)
    pop_metrics = summarize(*pop, catalog_size=len(movies), n_users=len(pop[0]))

    df = pd.DataFrame([
        {"model": "Two-Tower", **tt_metrics},
        {"model": "Popularity", **pop_metrics},
    ]).set_index("model")

    print("\nИТОГОВАЯ ТАБЛИЦА")
    print(df.to_string())

    out = ROOT / "reports" / "metrics.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out)
    print(f"\nСохранено: {out}")
    print(f"Общее время: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
