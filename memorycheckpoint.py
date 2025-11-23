from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
import re, json, uuid


# -------------------- STATE DEFINITIONS --------------------
class AgentState(TypedDict):
    task: str
    result: str
    score: int
    feedback: str
    iteration: int


# -------------------- CUSTOM CHECKPOINT SAVER --------------------
class IterationSaver(MemorySaver):
    """Custom in-memory saver that logs when checkpoints are saved."""

    def put(self, config, checkpoint, new_versions, metadata):
        print(f"[IterationSaver] ✅ Saved iteration checkpoint with metadata={metadata}")
        return super().put(config, checkpoint, new_versions, metadata)


memory_saver = IterationSaver()


# -------------------- HELPER: CREATE HYDRATED CHECKPOINT --------------------
from datetime import datetime

def make_checkpoint(state: AgentState):
    """Creates a fully hydrated checkpoint that persists correctly in MemorySaver."""
    return {
        "v": 1,  # version number required by LangGraph
        "ts": datetime.utcnow().isoformat(),  # timestamp for checkpoint trace
        "id": str(uuid.uuid4()),  # checkpoint id
        "channel_values": dict(state),  # ✅ full state snapshot
        "channel_versions": {k: str(uuid.uuid4()) for k in state.keys()},  # unique version per key
        "versions_seen": {"__input__": {}},  # helps preserve data when listing
        "updated_channels": list(state.keys()),
    }



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

    try:
        match = re.search(r"\{.*\}", review, re.DOTALL)
        data = json.loads(match.group()) if match else {"score": 5, "feedback": "Could not parse"}
    except Exception:
        data = {"score": 5, "feedback": "Parse error"}

    state.update(data)
    print("NODE 2 KE BAAD UPDATED STATE:", state)
    print("PRINTING SCORE-------", state["score"])

    config = {"configurable": {"thread_id": "task_001", "checkpoint_ns": "default"}}

    # ✅ Use hydrated checkpoint helper
    checkpoint = make_checkpoint(state)

    if state["score"] < 8 and state["iteration"] < 3:
        memory_saver.put(
            config,
            checkpoint,
            {k: checkpoint["channel_versions"][k] for k in checkpoint["channel_values"].keys()},
            {"source": "iteration_end", "completed_at": "reviewer"},
        )


    else:
        print("✅ Score sufficient or iteration limit reached — moving to human feedback.")

    return state


# -------------------- NODE 3: Human Feedback --------------------
def node3_human_feedback(state: AgentState):
    print("\n--- Human Feedback Node ---")
    print(f"Task: {state['task']}")
    print(f"Final Excel Formula:\n{state['result']}")
    print(f"Score: {state['score']}")
    print(f"Feedback: {state['feedback']}")
    print(f"Iterations done: {state['iteration']}")

    config = {"configurable": {"thread_id": "task_001", "checkpoint_ns": "default"}}
    checkpoint = make_checkpoint(state)

    memory_saver.put(
        config,
        checkpoint,
        {k: checkpoint["channel_versions"][k] for k in checkpoint["channel_values"].keys()},
        {"source": "iteration_end", "completed_at": "reviewer"},
    )

    return state


# -------------------- DECISION FUNCTION --------------------
def decide_next(s: AgentState):
    """Decide whether to loop or exit based on score/iteration."""
    try:
        if s["score"] >= 9 or s["iteration"] >= 3:
            print("➡️ decide_next(): Going to human_feedback")
            return "human_feedback"
        else:
            print("🔁 decide_next(): Going to generator")
            return "generator"
    except Exception as e:
        print(f"⚠️ decide_next() error: {e}")
        return "human_feedback"


# -------------------- GRAPH DEFINITION --------------------
workflow = StateGraph(AgentState)
workflow.add_node("generator", node1_generate)
workflow.add_node("reviewer", node2_review)
workflow.add_node("human_feedback", node3_human_feedback)

workflow.set_entry_point("generator")
workflow.add_edge("generator", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    decide_next,
    {"human_feedback": "human_feedback", "generator": "generator"},
)
workflow.add_edge("human_feedback", END)

app = workflow.compile(checkpointer=memory_saver)


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


# -------------------- INSPECT MEMORY CONTENTS --------------------
print("\n--- In-Memory Checkpoints ---")
checkpoints = list(memory_saver.list(config=config))
history = list(app.get_state_history(config))

# print("printing history length of default checkpoint", len(ls))
# print("printing history length of custom checkpoint", len(checkpoints))
# print("-" * 100)

# # Print one hydrated checkpoint to verify it contains full state
# print("✅ Custom checkpoint snapshot (hydrated):")
# print(checkpoints[-1].checkpoint)

print("\n================= GRAPH EXECUTION HISTORY =================")

history = list(app.get_state_history(config))

for i, h in enumerate(history, start=1):
    print(f"\n🧩 Checkpoint {i}:")
    print(f" Node: {h.metadata.get('source') or h.metadata.get('node') or 'unknown'}")
    print(f" Step: {h.metadata.get('step', 'N/A')}")
    print(f" Checkpoint ID: {getattr(h, 'id', 'N/A')}")
    print(f" Timestamp: {getattr(h, 'ts', 'N/A')}")

    state_values = getattr(h, "values", {})
    print(f" State Keys: {list(state_values.keys())}")

    if state_values:
        print(" Full State:")
        for k, v in state_values.items():
            snippet = (v[:200] + "...") if isinstance(v, str) and len(v) > 200 else v
            print(f"   {k}: {snippet}")
    else:
        print("   (No state values found)")

print("\n============================================================")




