from copy import deepcopy
from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

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


initial_state : StateType = {
    "iteration": 0,
    "visited": [],
    "run_c": False,
    "human_feedback": None,
    "node_a_output": None,
    "node_b_output": None,
    "node_c_output": None
}

config = {"configurable": {"thread_id": "demo-checkpoint"}}

def node_a(state):
    state["iteration"] += 1
    state["node_a_output"] = "Baah"
    state["visited"].append("A")
    return state

def node_b(state):
    state["visited"].append("B")
    state["node_b_output"] = f"Baah Baah #{state['iteration']}"
    state["run_c"] = (state["iteration"] % 3 == 0)
    return state

def node_c(state):
    state["visited"].append("C")
    state["node_c_output"] = f"BaahBaah Baah #{state['iteration']}"
    state["human_feedback"] = f"Feedback at iteration {state['iteration']}"
    return state

# Build graph
builder = StateGraph(StateType)
builder.add_node("A", node_a)
builder.add_node("B", node_b)
builder.add_node("C", node_c)
builder.set_entry_point("A")
builder.add_edge("A", "B")
builder.add_conditional_edges(
    "B",
    lambda s: "C" if s["run_c"] else "A",
    {"C": "C", "A": "A"}
)
builder.set_finish_point("C")

graph = builder.compile(checkpointer=memory)

# Run
final_state = graph.invoke(initial_state, config=config)
print("\n✅ Final State:", final_state)


history = list(graph.get_state_history(config))

print(history)

filtered_rows = []

for snap in history:
    # Skip snapshots with no state values
    if not snap.values:
        continue

    # Determine node name
    if snap.tasks:
        node_name = snap.tasks[0].name
    else:
        # Last snapshot after all nodes finished
        node_name = "__end__"

    state_values = snap.values

    # Capture Node-B only if run_c=False
    if node_name == "B" and state_values.get("run_c") is False:
        filtered_rows.append(state_values)

    # # Capture Node-C always
    # elif node_name == "C":
    #     filtered_rows.append(state_values)

    # Capture the last snapshot if tasks=() (post Node-C execution)
    elif node_name == "__end__":
        filtered_rows.append(state_values)

# Print rowwise full state
print("\n===== ROWWISE FULL STATE =====")
for idx, row in enumerate(filtered_rows, 1):
    print(f"\nRow {idx}:")
    for k, v in row.items():
        print(f"  {k}: {v}")

