from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    APP_NAME: str = "gRNA Predictor API"
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://*.github.io",
    ]
    MAX_SEQUENCE_LENGTH: int = 10000
    TOP_N_RESULTS: int = 5


settings = Settings()
