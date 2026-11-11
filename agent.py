from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import (
    AnyMessage, SystemMessage, ToolMessage, HumanMessage,
)
from langgraph.graph import StateGraph, START, END

load_dotenv()

model = init_chat_model("google_genai:gemini-3.5-flash-lite", timeout=30)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add `a` and `b`."""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` by `b`."""
    return a / b


tools = [add, multiply, divide]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def llm_call(state: AgentState):
    """The LLM decides whether to call a tool or answer."""
    system = SystemMessage(
        content="You are a helpful assistant that does arithmetic using the tools provided."
    )
    reply = model_with_tools.invoke([system] + state["messages"])
    return {"messages": [reply], "llm_calls": state.get("llm_calls", 0) + 1}


def tool_node(state: AgentState):
    """Run every tool the model asked for."""
    results = []
    for call in state["messages"][-1].tool_calls:
        observation = tools_by_name[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=str(observation), tool_call_id=call["id"]))
    return {"messages": results}


def should_continue(state: AgentState) -> Literal["tool_node", END]:
    if state["messages"][-1].tool_calls:
        return "tool_node"
    return END


builder = StateGraph(AgentState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
agent = builder.compile()


if __name__ == "__main__":
    result = agent.invoke({"messages": [HumanMessage(content="What is 12 times 5, plus 7?")]})
    for m in result["messages"]:
        m.pretty_print()
    print("LLM calls:", result["llm_calls"])