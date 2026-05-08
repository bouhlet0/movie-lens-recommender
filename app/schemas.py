from pydantic import BaseModel, Field
from typing import Literal, Annotated


# Request models
class RecommendRequest(BaseModel):
    input_type: Literal["titles", "imdb_ids"]
    items: Annotated[list[str], Field(min_length=1, max_length=50)]
    k: int = Field(default=10, ge=1, le=50)
    explore: bool = False
    pool_size: int = Field(default=75, ge=10, le=200)

    def model_post_init(self, __context) -> None:
        seen = set()
        deduped = []
        for item in self.items:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        self.items = deduped


# Response models
class MovieCard(BaseModel):
    rank: int
    title: str
    year: int | None
    genres: list[str]
    imdb_id: str | None


class RecommendResponse(BaseModel):
    recommendations: list[MovieCard]
    n_results: int
    input_type: Literal["titles", "imdb_ids", "csv_upload"]
    explore: bool


class SearchResult(BaseModel):
    title: str
    year: int | None
    genres: list[str]
    movie_id: int


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_users: int
    n_items: int