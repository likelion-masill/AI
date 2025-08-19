# app/schemas/common_response.py (예시)
from typing import Generic, Optional, TypeVar, Literal
from pydantic.generics import GenericModel

T = TypeVar("T")

class CommonResponse(GenericModel, Generic[T]):
    status: Literal["success", "error"]
    data: Optional[T] = None
    message: Optional[str] = None
