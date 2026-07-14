"""Configuration for the TuiML landing and documentation site."""

from datetime import datetime


class Settings:
    APP_NAME: str = "TuiML"
    APP_VERSION: str = "0.1.6"
    APP_STATUS: str = "Alpha"

    PROJECT_NAME: str = "TuiML"
    GITHUB_URL: str = "https://github.com/tuiml/tuiml"

    @property
    def COPYRIGHT_YEAR(self) -> int:
        return datetime.now().year


settings = Settings()
