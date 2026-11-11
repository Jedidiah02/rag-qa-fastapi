from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    number: int
    result: str


def handle_even(state: State):
    n = state["number"]
    return {"result": f"{n} is even, so half of it is {n // 2}"}


def handle_odd(state: State):
    n = state["number"]
    return {"result": f"{n} is odd, so triple plus one is {n * 3 + 1}"}

def handle_zero(state: State):
    n = state["number"]
    return {"result": "zero is special"}

def route(state: State):
    return "even" if state["number"] % 2 == 0 else "odd"

def route(state: State):
    if state["number"] == 0:
        return "zero"
    return "even" if state["number"] % 2 == 0 else "odd"


builder = StateGraph(State)
builder.add_node("even", handle_even)
builder.add_node("odd", handle_odd)
builder.add_node("zero", handle_zero)

builder.add_conditional_edges(START, route, {"even": "even", "odd": "odd", "zero": "zero"})
builder.add_edge("even", END)
builder.add_edge("odd", END)
builder.add_edge("zero", END)

graph = builder.compile()

print(graph.invoke({"number": 7}))
print(graph.invoke({"number": 10}))
print(graph.invoke({"number": 0}))