"""
Pydantic Schemas for the input events
"""

from typing import List
from pydantic import BaseModel


class Event(BaseModel):
    user_id: str
    timestamp: int
    features: List[float]

class EventBatch(BaseModel):
    events: List[Event]