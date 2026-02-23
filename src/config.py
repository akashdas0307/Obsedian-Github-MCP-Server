"""Application configuration loaded from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """All configuration for the MCP GitHub Server."""

    # --- GitHub ---
    github_repo_url: str = Field(
        ..., description="HTTPS clone URL for the target repository"
    )
    github_token: str = Field(
        ..., description="GitHub Personal Access Token with repo scope"
    )
    github_branch: str = Field(
        default="main", description="Branch to track"
    )

    # --- OAuth 2.1 ---
    oauth_client_id: str = Field(
        ..., description="OAuth client ID for Claude connector"
    )
    oauth_client_secret: str = Field(
        ..., description="OAuth client secret for Claude connector"
    )
    oauth_issuer_url: str = Field(
        ..., description="Public base URL of this server (JWT issuer)"
    )
    jwt_secret_key: str = Field(
        ..., description="Secret key for signing JWT tokens"
    )

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # --- Storage ---
    repo_dir: str = Field(
        default="/data/repo",
        description="Local path where the Git repo is cloned",
    )
    index_dir: str = Field(
        default="/data/index",
        description="Local path for FAISS index and metadata",
    )

    # --- Sync Timing ---
    pull_interval: int = Field(
        default=300, description="Seconds between periodic git pulls"
    )
    push_debounce: int = Field(
        default=120, description="Seconds to wait after last write before pushing"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # --- Derived helpers ---

    @property
    def repo_path(self) -> Path:
        return Path(self.repo_dir)

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)

    @property
    def authenticated_repo_url(self) -> str:
        """Insert token into HTTPS URL for git clone/push."""
        url = self.github_repo_url
        if url.startswith("https://"):
            return url.replace("https://", f"https://x-access-token:{self.github_token}@")
        return url


def get_settings() -> Settings:
    """Create and return a validated Settings instance."""
    return Settings()
