import re
import polars as pl
from inference import Recommender
from app.schemas import MovieCard, RecommendResponse, SearchResult, SearchResponse, HealthResponse

class RecommenderService:
    def __init__(self, recommender: Recommender) -> None:
        self.rec = recommender

    # Public API
    def recommend(
        self,
        input_type: str,
        items: list[str],
        k: int,
        explore: bool,
        pool_size: int,
    ) -> RecommendResponse:
        if input_type == "titles":
            df = self.rec.from_titles(items, k=k, wildcard=explore, pool_size=pool_size)
        else:
            imdb_ids = self._parse_imdb_ids(items)
            df = self.rec.from_imdb_ids(imdb_ids, k=k, wildcard=explore, pool_size=pool_size)

        recommendations = [
            self._row_to_movie_card(row) for row in df.iter_rows(named=True)
        ]

        return RecommendResponse(
            recommendations=recommendations,
            n_results=len(recommendations),
            input_type="csv_upload",
            explore=explore,
        )


    def recommend_from_csv(
        self,
        csv_path: str,
        min_rating: float,
        k: int,
        explore: bool,
        pool_size: int,
    ) -> RecommendResponse:
        df = self.rec.from_imdb_csv(
            csv_path,
            min_rating=min_rating,
            k=k,
            wildcard=explore,
            pool_size=pool_size,
        )

        recommendations = [
            self._row_to_movie_card(row) for row in df.iter_rows(named=True)
        ]

        return RecommendResponse(
            recommendations=recommendations,
            n_results=len(recommendations),
            input_type="imdb_ids",
            explore=explore,
        )



    def search(self, query: str, top_n: int = 5) -> SearchResponse:
        results_df = self.rec.search(query, top_n=top_n, min_score=60.0)

        results = [
            SearchResult(
                title=self._clean_title(row["title"]),
                year=self._extract_year(row["title"]),
                genres=self._parse_genres(row["genres"]),
                movie_id=int(row["movieId"]),
            )
            for row in results_df.iter_rows(named=True)
        ]

        return SearchResponse(results=results, query=query)

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=self.rec.model.is_loaded(),
            n_users=self.rec.model.n_users,
            n_items=self.rec.model.n_items,
        )

    
    # Private helpers
    def _row_to_movie_card(self, row: dict) -> MovieCard:
        raw_title = row["title"]
        imdb_url = row.get("imdb_url", "")
        imdb_id = self._extract_imdb_id(imdb_url) if imdb_url else None

        return MovieCard(
            rank=row["rank"],
            title=self._clean_title(raw_title),
            year=self._extract_year(raw_title),
            genres=self._parse_genres(row["genres"]),
            imdb_id=imdb_id,
        )

    @staticmethod
    def _parse_imdb_ids(items: list[str]) -> list[int]:
        ids = []
        for item in items:
            cleaned = item.strip().lstrip("tt")
            if cleaned.isdigit():
                ids.append(int(cleaned))
        return ids

    @staticmethod
    def _clean_title(title: str) -> str:
        # Remove trailing year
        return re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()

    @staticmethod
    def _extract_year(title: str) -> int | None:
        match = re.search(r'\((\d{4})\)\s*$', title)
        return int(match.group(1)) if match else None

    @staticmethod
    def _parse_genres(genres: str) -> list[str]:
        if not genres or genres == "(no genres listed)":
            return []
        return genres.split("|")

    @staticmethod
    def _extract_imdb_id(imdb_url: str) -> str | None:
        # "https://www.imdb.com/title/tt0110912" → "tt0110912"
        match = re.search(r'tt\d+', imdb_url)
        return match.group(0) if match else None