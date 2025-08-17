# app/schemas/common.py
from pydantic import BaseModel
from typing import Any, Optional

class CommonResponse(BaseModel):
    status: str  # "success" / "error"
    message: Optional[str] = None
    data: Optional[Any] = None
