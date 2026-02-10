from langgraph.graph import StateGraph, END

from agents.state import AgentState, add_trace
from agents.planner_agent import run as planner_run
from agents.retriever_agent import run as retriever_run
from agents.writer_agent import run as writer_run


def planner_node(state: AgentState) -> AgentState:
    return planner_run(state)


def retriever_node(state: AgentState) -> AgentState:
    return retriever_run(state)


def writer_node(state: AgentState) -> AgentState:
    return writer_run(state)


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


def run(task: str, top_k: int = 5) -> AgentState:
    app = build_graph()
    state: AgentState = {
        "task": task,
        "top_k": top_k,
        "trace": [],
    }
    add_trace(state, "system", "start", "Starting LangGraph run")
    out = app.invoke(state)
    add_trace(out, "system", "end", "Finished LangGraph run")
    return out
