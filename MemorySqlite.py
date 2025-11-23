from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict
import sqlite3
import re
import json

# -------------------- STATE DEFINITIONS --------------------
class AgentState(TypedDict):
    task: str
    result: str
    score: int
    feedback: str
    iteration: int

# -------------------- NODE 1: Generator --------------------
def node1_generate(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = f"Generate an Excel formula for: {state['task']} and fill dummy data."
    excel_result = llm.invoke(prompt)
    state["result"] = excel_result.content
    state["iteration"] += 1
    return state

# -------------------- NODE 2: Reviewer --------------------
def node2_review(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    review_prompt = (
        f"Review this Excel work and score (1–10). "
        f"Output JSON: {{'score': x, 'feedback': '...'}}\n\nWork:\n{state['result']}"
    )
    review = llm.invoke(review_prompt).content

    # Parse JSON from LLM
    try:
        match = re.search(r"\{.*\}", review, re.DOTALL)
        data = json.loads(match.group()) if match else {"score": 5, "feedback": "Could not parse"}
    except:
        data = {"score": 5, "feedback": "Parse error"}

    state.update(data)

    # Decide next node
    if state["score"] < 8 and state["iteration"] < 3:
        next_node = "generator"
    else:
        next_node = "human_feedback"

    # Return state update and next node
    return {"update": dict(state), "goto": next_node}

# -------------------- NODE 3: Human Feedback --------------------
def node3_human_feedback(state: AgentState):
    print("\n--- Human Feedback Node ---")
    print(f"Task: {state['task']}")
    print(f"Final Excel Formula:\n{state['result']}")
    print(f"Score: {state['score']}")
    print(f"Feedback: {state['feedback']}")
    print(f"Iterations done: {state['iteration']}")
    return state

# -------------------- GRAPH DEFINITION --------------------
workflow = StateGraph(AgentState)

workflow.add_node("generator", node1_generate)
workflow.add_node("reviewer", node2_review)
workflow.add_node("human_feedback", node3_human_feedback)

workflow.set_entry_point("generator")
workflow.add_edge("generator", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    lambda state: (
        "generator" if state["score"] < 8 and state["iteration"] < 3 else "human_feedback"
    ),
)
workflow.add_edge("human_feedback", END)

# -------------------- SQLITE CHECKPOINT --------------------
db_path = "agent_checkpoint.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
sqlite_saver = SqliteSaver(conn)

app = workflow.compile(checkpointer=sqlite_saver)

# -------------------- RUN AGENT --------------------
initial_state: AgentState = {
    "task": "Calculate compound interest",
    "result": "",
    "score": 0,
    "feedback": "",
    "iteration": 0,
}

config = {"configurable": {"thread_id": "task_001"}}

final_state = app.invoke(initial_state, config=config)
print("\n--- Checkpoint History ---")

# Materialize before the cursor closes
checkpoints = list(app.checkpointer.list(config=config))

l = app.get_state(config)
print("get_state", l)
ls = list(app.get_state_history(config))
print("printing history", ls)
# for i, cp in enumerate(checkpoints, start=1):
#     print(f"\n--- Checkpoint {i} ---")

#     # cp is a CheckpointTuple, inspect its parts
#     print(f"ID: {cp.checkpoint.get('id')}")
#     print(f"Timestamp: {cp.checkpoint.get('ts')}")
#     print(f"Source: {cp.metadata.get('source')} | Step: {cp.metadata.get('step')}")

#     # Channel (state) values actually live under 'channel_values'
#     channel_values = cp.checkpoint.get("channel_values", {})
#     if channel_values:
#         for k, v in channel_values.items():
#             # trim long text for readability
#             snippet = (v[:200] + "...") if isinstance(v, str) and len(v) > 200 else v
#             print(f"{k}: {snippet}")
#     else:
#         print("(No channel values found)")
