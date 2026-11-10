"""
Загрузка MovieLens 100k для Two-Tower.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(
        RAW_DIR / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )


def load_movies() -> pd.DataFrame:
    cols = ["movie_id", "title", "release_date", "video_date", "imdb_url"] + GENRES
    df = pd.read_csv(
        RAW_DIR / "u.item",
        sep="|",
        names=cols,
        encoding="latin-1",
    )
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["year"] = df["year"].fillna(1990).astype("int32")

    def genres_str(row):
        return " ".join(g for g in GENRES if row[g] == 1)

    df["genres"] = df.apply(genres_str, axis=1)
    df["movie_id"] = df["movie_id"].astype(str)
    df["title"] = df["title"].fillna("")
    return df[["movie_id", "title", "genres", "year"]]


def load_users() -> pd.DataFrame:
    df = pd.read_csv(
        RAW_DIR / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
    )
    df["user_id"] = df["user_id"].astype(str)
    return df[["user_id", "age", "gender", "occupation"]]


def build_user_features() -> pd.DataFrame:
    return load_users()


def build_movie_features() -> pd.DataFrame:
    return load_movies()


def build_interactions(min_rating: int = 4) -> pd.DataFrame:
    """
    Позитивные взаимодействия для retrieval.
    ВАЖНО: rating сохраняется, чтобы при обучении можно было
    выбирать один позитив на пользователя и не создавать
    false negatives внутри in-batch negatives.
    """
    ratings = load_ratings()
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["movie_id"] = ratings["movie_id"].astype(str)
    return ratings.loc[
        ratings["rating"] >= min_rating,
        ["user_id", "movie_id", "rating"]
    ].reset_index(drop=True)


def build_all_interactions() -> pd.DataFrame:
    ratings = load_ratings()
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["movie_id"] = ratings["movie_id"].astype(str)
    return ratings[["user_id", "movie_id", "rating"]].reset_index(drop=True)


def split_users(interactions: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42):
    """Единый user-level split для train/evaluate."""
    users = pd.Series(interactions["user_id"].unique())
    shuffled = users.sample(frac=1.0, random_state=seed).to_numpy()
    n_test = int(len(shuffled) * test_fraction)
    test_users = set(shuffled[:n_test])
    train_users = set(shuffled[n_test:])
    return train_users, test_users


def save_processed():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    build_user_features().to_csv(PROCESSED_DIR / "users_features.csv", index=False)
    build_movie_features().to_csv(PROCESSED_DIR / "movies_features.csv", index=False)
    build_interactions().to_csv(PROCESSED_DIR / "interactions_pos.csv", index=False)
    build_all_interactions().to_csv(PROCESSED_DIR / "interactions_all.csv", index=False)
