from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    number: int
    result: str


def check_number(state: State):
    return {"result": "even" if state["number"] % 2 == 0 else "odd"}


def shout(state: State):
    return {"result": state["result"].upper() + "!"}

def add_message(state: State):
    return {"result": f"{state['number']} is {state['result']}"}


builder = StateGraph(State)
builder.add_node("check", check_number)
builder.add_node("shout", shout)
builder.add_node("add", add_message)
builder.add_edge(START, "check")
builder.add_edge("check", "shout")
builder.add_edge("shout", "add")
builder.add_edge("add", END)

graph = builder.compile()
print(graph.invoke({"number": 7}))