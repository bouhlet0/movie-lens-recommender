import tempfile
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from inference import Recommender
from app.schemas import RecommendRequest, RecommendResponse, SearchResponse, HealthResponse
from app.service import RecommenderService


# Config
MODEL_PATH      = "experiments/results/lightgcn_model_256dim_150epoch.pkl"
ARTIFACTS_PATH  = "experiments/results/inference_artifacts.pkl"
DATA_DIR        = Path("data/ml-32m")


# Lifespan: load model once at startup
service: RecommenderService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    print("Loading model and artifacts...")
    recommender = Recommender(
        model_path=MODEL_PATH,
        artifacts_path=ARTIFACTS_PATH,
        data_dir=DATA_DIR,
    )
    service = RecommenderService(recommender)
    print("Model loaded. Ready to recommend.")
    yield
    print("Shutting down.")


# App
app = FastAPI(
    title="Movie Recommender API",
    description="LightGCN-based collaborative filtering recommender trained on MovieLens 32M.",
    version="0.0.1",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Routes
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def root():
    return FileResponse("app/static/index.html")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    return service.health()


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search(q: str = Query(..., min_length=1, max_length=100)):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    return service.search(query=q, top_n=7)


@app.post("/recommend", response_model=RecommendResponse, tags=["Inference"])
async def recommend(request: RecommendRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    try:
        return service.recommend(
            input_type=request.input_type,
            items=request.items,
            k=request.k,
            explore=request.explore,
            pool_size=request.pool_size,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/upload", response_model=RecommendResponse, tags=["Inference"])
async def upload(
    file: UploadFile = File(...),
    min_rating: float = Query(default=7.0, ge=1.0, le=10.0),
    k: int = Query(default=10, ge=1, le=50),
    explore: bool = Query(default=False),
    pool_size: int = Query(default=75, ge=10, le=200),
):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Write upload to a temp file since Recommender expects a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        return service.recommend_from_csv(
            csv_path=tmp_path,
            min_rating=min_rating,
            k=k,
            explore=explore,
            pool_size=pool_size,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)