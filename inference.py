import pickle
import numpy as np
import polars as pl
from pathlib import Path
from rapidfuzz import process, fuzz
import re

from models.lightgcn import LightGCNModel


class Recommender:
    def __init__(
        self,
        model_path: str,
        artifacts_path: str,
        data_dir: Path = Path("data/ml-32m"),
    ) -> None:
        self.model = LightGCNModel.load(model_path)
        self.links = pl.read_csv(data_dir / "links.csv")
        self.movies = pl.read_csv(data_dir / "movies.csv")

        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)

        self.item2idx: dict[int, int] = artifacts["item2idx"]
        self.idx2item: dict[int, int] = artifacts["idx2item"]

    def _imdb_ids_to_item_indices(self, imdb_ids: list[int]) -> list[int]:
        matched = self.links.filter(pl.col("imdbId").is_in(imdb_ids))
        movie_ids = matched["movieId"].to_list()
        indices = [self.item2idx.get(mid) for mid in movie_ids]
        return [i for i in indices if i is not None]

    def _title_search(self, title: str, top_n: int = 5, min_score: float = 60.0) -> pl.DataFrame:
        raw_titles = self.movies["title"].to_list()
        normalized_titles = [self._normalize_title(t) for t in raw_titles]
        normalized_query = self._normalize_title(title)

        matches = process.extract(
            normalized_query, normalized_titles, scorer=fuzz.WRatio, limit=top_n
        )
        matched_titles = [raw_titles[normalized_titles.index(m[0])] for m in matches if m[1] >= min_score]
        if not matched_titles:
            return self.movies.head(0)
        return self.movies.filter(pl.col("title").is_in(matched_titles))

    def from_imdb_ids(self, imdb_ids: list[int], k: int = 10) -> pl.DataFrame:
        item_indices = self._imdb_ids_to_item_indices(imdb_ids)
        return self._recommend(item_indices, k)

    def from_imdb_csv(self, csv_path: str, min_rating: float = 7.0, k: int = 10) -> pl.DataFrame:
        imdb_df = pl.read_csv(csv_path)
        filtered = imdb_df.filter(pl.col("Your Rating") >= min_rating)
        imdb_ids = (
            filtered["Const"]
            .str.strip_chars("tt")
            .cast(pl.Int64)
            .to_list()
        )
        ratings = filtered["Your Rating"].to_list()
        item_indices, weights = self._imdb_ids_to_item_indices_weighted(imdb_ids, ratings)
        return self._recommend(item_indices, k, weights=weights)

    def from_titles(self, titles: list[str], k: int = 10) -> pl.DataFrame:
        item_indices = []
        for title in titles:
            results = self._title_search(title, top_n=1, min_score=60.0)
            if len(results) == 0:
                print(f"  No match found for: '{title}'")
                continue
            movie_id = int(results["movieId"][0])
            idx = self.item2idx.get(movie_id)
            if idx is not None:
                item_indices.append(idx)
                print(f"  '{title}' → {results['title'][0]}")
            else:
                print(f"  '{title}' matched but not in training data")

        if not item_indices:
            print("  No valid items found, returning empty recommendations.")
            return pl.DataFrame(schema={"rank": pl.Int64, "title": pl.Utf8, "genres": pl.Utf8, "imdb_url": pl.Utf8})

        return self._recommend(item_indices, k)

    def _recommend(self, item_indices: list[int], k: int, weights: list[float] | None = None) -> pl.DataFrame:
        if not item_indices:
            return pl.DataFrame(schema={"rank": pl.Int64, "title": pl.Utf8, "genres": pl.Utf8, "imdb_url": pl.Utf8})

        factors = self.model._item_factors[item_indices]

        if weights is not None and len(weights) == len(item_indices):
            w = np.array(weights, dtype=np.float32)
            w = w / w.sum()
            proxy_user = (factors * w[:, None]).sum(axis=0)
        else:
            proxy_user = factors.mean(axis=0)

        scores = self.model._item_factors @ proxy_user
        scores[item_indices] = -np.inf

        top_k = np.argsort(-scores)[:k]
        top_movie_ids = [self.idx2item[i] for i in top_k]

        movies_out = (
            self.movies
            .filter(pl.col("movieId").is_in(top_movie_ids))
            .join(
                self.links.select(["movieId", "imdbId"]),
                on="movieId",
                how="left",
            )
            .with_columns(
                pl.concat_str(
                    pl.lit("https://www.imdb.com/title/tt"),
                    pl.col("imdbId").cast(pl.Utf8).str.zfill(7),
                ).alias("imdb_url")
            )
            .select(["movieId", "title", "genres", "imdb_url"])
        )

        order = pl.DataFrame({
            "movieId": top_movie_ids,
            "rank": list(range(1, k + 1)),
        })

        return (
            order
            .join(movies_out, on="movieId", how="left")
            .select(["rank", "title", "genres", "imdb_url"])
        )
        
    def _imdb_ids_to_item_indices_weighted(self, imdb_ids: list[int], ratings: list[float]) -> tuple[list[int], list[float]]:
        matched = self.links.filter(pl.col("imdbId").is_in(imdb_ids))
        valid_indices = []
        valid_weights = []
        for row in matched.iter_rows(named=True):
            idx = self.item2idx.get(row["movieId"])
            # find corresponding rating by matching back to imdb_id
            imdb_pos = imdb_ids.index(row["imdbId"])
            if idx is not None:
                valid_indices.append(idx)
                valid_weights.append(ratings[imdb_pos])
        return valid_indices, valid_weights
            
    @staticmethod
    def _normalize_title(title: str) -> str:
        title = title.lower().strip()
        title = re.sub(r'\(\d{4}\)', '', title).strip()
        title = re.sub(r'^(the|a|an)\s+', '', title).strip()
        title = re.sub(r',\s*(the|a|an)$', '', title).strip()
        return title