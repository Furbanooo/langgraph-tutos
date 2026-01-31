'''Extend the agent in [module-1/agent.ipynb](../module-1/agent.ipynb) by planning a new capability (tool or sub-flow). Describe the input/output contract, where it should be called, and how it changes the user experience.'''

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from langchainacademy_module_practrices.module01.practrice1 import state

load_dotenv()
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

def subtract(a: int, b: int) -> int:
    """Subtracts b from a.

    Args:
        a: first int
        b: second int
    """
    return a - b

def currencies_conversion(amount: float, from_currency: str, to_currency: str) -> float:
    """Converts amount from one currency to another.

    Args:
        amount: amount to convert
        from_currency: currency to convert from
        to_currency: currency to convert to
    """
    # Dummy implementation for example purposes
    conversion_rates = {
        ("USD", "EUR"): 0.85,
        ("EUR", "USD"): 1.18,
        ("USD", "JPY"): 110.0,
        ("JPY", "USD"): 0.0091,
    }
    rate = conversion_rates.get((from_currency, to_currency), 1)
    return amount * rate

tools = [add, multiply, divide, subtract, currencies_conversion]

# Define LLM with bound tools
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant that performs arithmetic and currency conversions on user inputs. Detect when the user is multiplying numbers or converting currencies, and use the appropriate tool to help with the request.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()

while True:
    user_input = input("Enter your request (or type 'exit' to quit): ")
    if user_input == "exit":
        break

    state["messages"].append(HumanMessage(content=user_input))
    messages = graph.invoke({"messages": state["messages"]})
    for m in messages['messages']:
        m.pretty_print()

