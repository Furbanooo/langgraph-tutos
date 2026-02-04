"""Configure an external memory checkpoint in your graph and persistence across runs."""

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import END, MessagesState, StateGraph


load_dotenv()

db_path = "/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)

model = ChatOpenAI(model="gpt-4o", temperature=0)


class ChatState(MessagesState):
	pass


SYSTEM_MESSAGE = SystemMessage(
	content="You are a concise, helpful chatbot. Respond clearly and briefly."
)


def chat_node(state: ChatState) -> dict:
	messages = [SYSTEM_MESSAGE] + state["messages"]
	response = model.invoke(messages)
	return {"messages": [AIMessage(content=response.content)]}


def build_graph():
	builder = StateGraph(ChatState)
	builder.add_node("chat", chat_node)
	builder.set_entry_point("chat")
	builder.add_edge("chat", END)
	return builder.compile(checkpointer=memory)


def run_chatbot() -> None:
	graph = build_graph()
	state: ChatState = {"messages": []}
	print("Chatbot ready. Type 'exit' to quit.")
	while True:
		user_input = input("You: ").strip()
		if user_input.lower() == "exit":
			print("Chatbot: Goodbye!")
			break

		state["messages"].append(HumanMessage(content=user_input))
		state = graph.invoke(state)
		last_message = state["messages"][-1]
		if isinstance(last_message, AIMessage):
			print(f"Chatbot: {last_message.content}")


if __name__ == "__main__":
	run_chatbot()