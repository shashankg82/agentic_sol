from copy import deepcopy
from operator import add
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import InMemorySaver

memory = MemorySaver()



class StateType(TypedDict):
    iteration: int
    # visited: Annotated[list[str], add]
    visited: List[str]
    run_c: bool
    human_feedback: str
    node_a_output: str
    node_b_output: str
    node_c_output: str

# ✅ real initial data (not class)
initial_state: StateType = {
    "iteration": 0,
    "visited": [],
    "run_c": False,
    "human_feedback": "",
    "node_a_output": "",
    "node_b_output": "",
    "node_c_output": ""
}

config = {"configurable": {"thread_id": "demo-checkpoint"}}

def node_a(state):
    state["iteration"] += 1
    state["node_a_output"] = "Baah"
    state["visited"] = state["visited"] + ["A"]
    return state

def node_b(state):
    state["visited"] = state["visited"] + ["B"]
    state["node_b_output"] = f"Baah Baah #{state['iteration']}"
    state["run_c"] = (state["iteration"] % 3 == 0)
    return state

def node_c(state):
    state["visited"] = state["visited"] + ["C"]
    state["node_c_output"] = f"BaahBaah Baah #{state['iteration']}"
    state["human_feedback"] = f"Feedback at iteration {state['iteration']}"
    return state

# Build graph
# builder = StateGraph(dict)
builder = StateGraph(StateType)
builder.add_node("A", node_a)
builder.add_node("B", node_b)
builder.add_node("C", node_c)
builder.set_entry_point("A")
# builder.add_edge(START, "A")
builder.add_edge("A", "B")
builder.add_conditional_edges(
    "B",
    lambda s: "C" if s["run_c"] else "A",
    {"C": "C", "A": "A"}
)
# builder.add_edge("C", END)
builder.set_finish_point("C")

graph = builder.compile(checkpointer=memory)

# Run
final_state = graph.invoke(initial_state, config=config)
print("\n✅ Final State:", final_state)

history = list(graph.get_state_history(config))

print(history)


filtered_rows = []
previous_next = None

# history is newest→oldest, so reversed() gives oldest→newest
for snap in reversed(history):
    if not snap.values:
        continue

    state_values = snap.values

    if previous_next:
        node_name = previous_next[0]
    else:
        node_name = "__start__"

    if node_name == "B" and state_values.get("run_c") is False:
        filtered_rows.append(state_values)
    elif node_name == "C" or node_name == "__end__":
        filtered_rows.append(state_values)

    previous_next = snap.next

# No need to reverse again
print("\n===== ROWWISE FULL STATE =====")
for idx, row in enumerate(filtered_rows, 1):
    print(f"\nRow {idx}:")
    for k, v in row.items():
        print(f"  {k}: {v}")




