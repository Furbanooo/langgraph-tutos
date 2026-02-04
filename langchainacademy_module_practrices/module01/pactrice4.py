''' Add a state field to represent “decision rationale.” Define what data it should hold, when it updates, and how it helps debugging or observability.'''

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model("gpt-3.5-turbo", temperature=0)

class State(MessagesState):
    decision_rationale: str = ""

class decisionRationale(BaseModel):
    decision_rationale: bool = Field(
        ...,
        description="the fact entered by the user are clasified iether rational or not in a boolean format"
    )

def decision_rationale_node(state: State):
    last_message = state["messages"][-1]

    system_prompt = SystemMessage(content="You are an expert assistant that detect whether the facts entered by the user are classified as rational or not in a boolean format.")

    messages = [
        system_prompt,
        HumanMessage(content='Classify the following facts as rational (true) or not rational (false): ' + last_message.content)
    ]

    response = llm.invoke(messages)
    return {"decision_rationale": response.content}

builder = StateGraph(State)
builder.add_node("decision_rationale_node", decision_rationale_node)
builder.add_edge(START, "decision_rationale_node") 
builder.add_edge("decision_rationale_node", END)
graph = builder.compile()

def runRationalBot():
    state: State = {
        "messages": [],
        "decision_rationale": ""
    }
    while True:
        user_input = input("Enter facts to classify as rational or not (or 'exit' to quit): ")
        if user_input == 'exit':
            break

        state['messages'].append(HumanMessage(content=user_input))
        state= graph.invoke(state)
        print("Decision Rationale:", state["decision_rationale"])

if __name__ == "__main__":
    runRationalBot()
