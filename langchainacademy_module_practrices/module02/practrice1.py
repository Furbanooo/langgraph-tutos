"""Define a schema with both short-term messages and long-term memory fields."""

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class MemoryTypeSchema(BaseModel):
    short_term_messages: list[BaseMessage] = Field(
        ...,
        description="Recent conversation messages (short-term)."
    )
    long_term_memory: str = Field(
        ...,
        description="Key facts or information retained over time."
    )

