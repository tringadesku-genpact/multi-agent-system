from langgraph.graph import StateGraph, END

from agents.state import AgentState, add_trace
from agents.planner_agent import run as planner_run
from agents.retriever_agent import run as retriever_run
from agents.writer_agent import run as writer_run
from agents.verifier_agent import run as verifier_run

from agents.persistence import save_run
from agents.guardrails_agent import run as guardrails_run


def planner_node(state: AgentState) -> AgentState:
    return planner_run(state)


def retriever_node(state: AgentState) -> AgentState:
    return retriever_run(state)


def writer_node(state: AgentState) -> AgentState:
    return writer_run(state)


def verifier_node(state: AgentState) -> AgentState:
    return verifier_run(state)


def guardrails_node(state: AgentState) -> AgentState:
    return guardrails_run(state)


def _route_after_guardrails(state: AgentState):
    # If guardrails blocked the request, end immediately
    # print("ROUTE AFTER GUARDRAILS stop=", state.get("stop"))
    return END if state.get("stop") else "planner"


def _route_after_verifier(state: AgentState):
    # If verifier requests retry, go back to retriever; else finish.
    return "retriever" if state.get("needs_retry") else END


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("guardrails", guardrails_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("writer", writer_node)
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("guardrails")

    # Guardrails decides whether we continue or stop
    graph.add_conditional_edges("guardrails", _route_after_guardrails, ["planner", END])

    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "writer")
    graph.add_edge("writer", "verifier")

    # conditional edge (loop once if needed)
    graph.add_conditional_edges("verifier", _route_after_verifier, ["retriever", END])

    return graph.compile()


def run(task: str, top_k: int = 5) -> AgentState:
    app = build_graph()
    state: AgentState = {
        "task": task,
        "top_k": top_k,
        "trace": [],
        "retried": False,
        "needs_retry": False,
    }

    add_trace(state, "system", "start", "Starting LangGraph run")
    out = app.invoke(state)
    add_trace(out, "system", "end", "Finished LangGraph run")
    save_run(out)
    return out
