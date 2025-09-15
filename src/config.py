from pydantic import BaseSettings

class Config(BaseSettings):
    informatics_server_url: str = "http://localhost:8080"

    class Config:
        env_file = ".env"

config = Config()