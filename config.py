from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: str = "openai"
    chat_model: str = "gpt-4o"
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings(): return Settings()
