# Design a mini "routing assistant" project: define the user roles, the intents you will support, and the criteria for routing between nodes. Sketch the graph and explain the decision logic in words.(https://github.com/Furbanooo/langchain-academy/blob/main/module-1/PRACTICE.md)

from os import system
from dotenv import load_dotenv
from typing import TypedDict
from pydantic import BaseModel, Field
from pydantic import Field
from typing_extensions import Annotated, Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load environment variables from .env file
load_dotenv()
# Initialize the chat model
llm = init_chat_model("gpt-4", temperature=0)

# Define the state schemas
class state(TypedDict):
    messages: Annotated[list, add_messages]
    problem_type: str | None

class problemClassifier(BaseModel):
    problem_type: Literal["software", "hardware", "unclear"] = Field(
        ...,
        description="Classify if the technical issue is related to software or hardware."
    )

#agent instructions
sys_msg = SystemMessage(content= '''You are a helpful technical support agent. identify whether the user's issue is related to software or hardware and return the appropriate classification. If the issue is unclear, try to collect more information from the user to clarify the problem.''')

software_msg = SystemMessage(content= '''You are a software support agent. Provide assistance with software-related issues, such as installation problems, bugs, or usage questions. Be clear and concise in your explanations.''')

hardware_msg = SystemMessage(content= '''You are a hardware support agent. Provide assistance with hardware-related issues, such as device malfunctions, connectivity problems, or physical repairs. Be clear and concise in your explanations.''')

#Define the functions(nodes)
def classify_problem(state: state):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(problemClassifier)

    result = classifier_llm.invoke([
        sys_msg,
        HumanMessage(content= last_message.content)
    ])
    return {"problem_type": result.problem_type} 



def router(state: state):
    problem_type = state.get("problem_type")
    if problem_type == "software":
        return {"next": "software"} 
    elif problem_type == "hardware":
        return {"next": "hardware"}
    else:
        return {"next": "unclear"}


def software_agent(state: state):
    last_message = state["messages"][-1]

    messages = [
        software_msg,
        HumanMessage(content= last_message.content)
    ]

    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}

def hardware_agent(state: state):
    last_message = state["messages"][-1]

    messages = [
        hardware_msg,
        HumanMessage(content= last_message.content)
    ]

    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}

def clarify_agent(state: state):
    last_message = state["messages"][-1]

    messages = [
        SystemMessage(content="Ask a brief follow-up question to clarify whether this is a software or hardware issue."),
        HumanMessage(content=last_message.content)
    ]

    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}

# Define the graph
builder = StateGraph(state)

# Add nodes to the graph
builder.add_node("classify_problem", classify_problem)
builder.add_node("router", router)
builder.add_node("software_agent", software_agent)
builder.add_node("hardware_agent", hardware_agent)
builder.add_node("clarify_agent", clarify_agent)

# Define edges between nodes
builder.add_edge(START, "classify_problem")
builder.add_edge("classify_problem", "router")
builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"software": "software_agent", 
     "hardware": "hardware_agent", 
     "unclear": "clarify_agent"}
)
builder.add_edge("software_agent", END)
builder.add_edge("hardware_agent", END) 
builder.add_edge("clarify_agent", END)
graph = builder.compile()

def run_technical_support_bot():
    state = {"messages": [], "problem_type": None}

    while True:
        user_input = input("Describe your technical issue (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Thank you for using the technical support bot. Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = graph.invoke(state)
        if state.get("messages"):
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage):
                print(f"Support Bot: {last_message.content}")
            else:
                print("Support Bot: I'm sorry, I couldn't process your request.")
        else:
            print("Support Bot: I'm sorry, I couldn't process your request.")
    

if __name__ == "__main__":
    run_technical_support_bot()