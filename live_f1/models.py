from typing import Optional
from pydantic import BaseModel


class ReplayConfig(BaseModel):
    year: int
    race: str
    session: str = "R"
    speed: float = 1.0
    start_at: float = 0.0