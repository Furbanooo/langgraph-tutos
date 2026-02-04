"""Implement a reducer that merges messages safely and test it with two updates."""

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


def message_reducer(
    left: list[BaseMessage] | None,
    right: list[BaseMessage] | None,
) -> list[BaseMessage]:
    if not left:
        left = []
    if not right:
        right = []
    return left + right


first_update = [HumanMessage(content="Hi")]
second_update = [AIMessage(content="Hello!")]
combined = message_reducer(first_update, second_update)
print([m.content for m in combined])

