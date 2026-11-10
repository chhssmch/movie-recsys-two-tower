"""
Two-Tower модель для cold-start рекомендаций.

Сценарий проекта:
- user tower использует только признаки пользователя;
- user_id НЕ используется как обучаемый признак, иначе cold-start
  для unseen users превращается в OOV/untrained embedding;
- item tower использует movie_id + жанры + год;
- обучение — retrieval с in-batch negatives.
"""

import tensorflow as tf
import tensorflow_recommenders as tfrs


class UserTower(tf.keras.Model):
    def __init__(self, users_df, embedding_dim=64):
        super().__init__()

        self.age_normalizer = tf.keras.layers.Normalization(axis=None)
        self.age_normalizer.adapt(users_df["age"].to_numpy(dtype="float32"))

        self.gender_lookup = tf.keras.layers.StringLookup(
            vocabulary=sorted(users_df["gender"].astype(str).unique().tolist()),
            mask_token=None,
            num_oov_indices=1,
        )
        self.gender_vocab_size = self.gender_lookup.vocabulary_size()

        self.occupation_lookup = tf.keras.layers.StringLookup(
            vocabulary=sorted(users_df["occupation"].astype(str).unique().tolist()),
            mask_token=None,
            num_oov_indices=1,
        )
        self.occupation_embedding = tf.keras.layers.Embedding(
            self.occupation_lookup.vocabulary_size(),
            16,
        )

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(embedding_dim),
        ])

    def call(self, inputs):
        age = tf.cast(inputs["age"], tf.float32)
        gender = inputs["gender"]
        occupation = inputs["occupation"]

        gender_one_hot = tf.one_hot(
            self.gender_lookup(gender),
            depth=self.gender_vocab_size,
        )

        x = tf.concat([
            tf.expand_dims(self.age_normalizer(age), -1),
            tf.cast(gender_one_hot, tf.float32),
            self.occupation_embedding(self.occupation_lookup(occupation)),
        ], axis=1)

        return self.dense(x)


class ItemTower(tf.keras.Model):
    def __init__(self, movies_df, embedding_dim=64):
        super().__init__()

        self.movie_id_lookup = tf.keras.layers.StringLookup(
            vocabulary=movies_df["movie_id"].astype(str).unique().tolist(),
            mask_token=None,
            num_oov_indices=1,
        )
        self.movie_id_embedding = tf.keras.layers.Embedding(
            self.movie_id_lookup.vocabulary_size(),
            embedding_dim,
        )

        self.genres_lookup = tf.keras.layers.TextVectorization(
            output_mode="multi_hot",
            standardize="lower_and_strip_punctuation",
        )
        self.genres_lookup.adapt(movies_df["genres"].astype(str).to_numpy())

        self.year_normalizer = tf.keras.layers.Normalization(axis=None)
        self.year_normalizer.adapt(movies_df["year"].to_numpy(dtype="float32"))

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(embedding_dim),
        ])

    def call(self, inputs):
        movie_id = inputs["movie_id"]
        genres = inputs["genres"]
        year = tf.cast(inputs["year"], tf.float32)

        x = tf.concat([
            self.movie_id_embedding(self.movie_id_lookup(movie_id)),
            tf.cast(self.genres_lookup(genres), tf.float32),
            tf.expand_dims(self.year_normalizer(year), -1),
        ], axis=1)

        return self.dense(x)


class TwoTowerModel(tfrs.Model):
    def __init__(self, user_tower, item_tower, movies_df):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

        movie_ds = tf.data.Dataset.from_tensor_slices({
            "movie_id": tf.constant(movies_df["movie_id"].astype(str).values),
            "genres": tf.constant(movies_df["genres"].astype(str).values),
            "year": tf.constant(movies_df["year"].values, dtype=tf.int32),
        }).batch(256)

        # FactorizedTopK используется только как offline training metric.
        # Для production retrieval индекс строится отдельно.
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movie_ds.map(
                    lambda f: (
                        f["movie_id"],
                        self.item_tower(f),
                    )
                )
            )
        )

    def call(self, features, training=False):
        user_emb = self.user_tower({
            "user_id": features["user_id"],  # совместимость с dataset
            "age": features["age"],
            "gender": features["gender"],
            "occupation": features["occupation"],
        })
        item_emb = self.item_tower({
            "movie_id": features["movie_id"],
            "genres": features["genres"],
            "year": features["year"],
        })
        return user_emb, item_emb

    def compute_loss(self, features, training=False):
        user_emb, item_emb = self(features, training=training)
        return self.task(
            user_emb,
            item_emb,
            compute_metrics=not training,
        )


def build_model(users_df, movies_df, embedding_dim=64):
    return TwoTowerModel(
        UserTower(users_df, embedding_dim),
        ItemTower(movies_df, embedding_dim),
        movies_df,
    )
