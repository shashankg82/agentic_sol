"""
Agentic FAISS pipeline (per-iteration rows) — compatible with langgraph v0.6.x
- MemorySaver checkpointing
- FAISS per-iteration vectors + metadata JSON
- sentence-transformers embeddings by default
- Uses OpenAI API if OPENAI_API_KEY is set; otherwise a small deterministic fallback
"""

import os
import json
import time
import faiss
import numpy as np
from typing import Dict, Any, List, Tuple
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
# Optional OpenAI import (only used if OPENAI_API_KEY set)
try:
    import openai
except Exception:
    openai = None

# sentence-transformers for local embeddings
from sentence_transformers import SentenceTransformer


LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)
# ---------------------------
# Flexible langgraph imports
# ---------------------------
_graph_err = None
Graph = None
State = None
MemorySaver = None

try:
    # try a few common import locations for Graph/State/MemorySaver
    try:
        from langgraph import StateGraph as Graph  # some versions
    except Exception:
        try:
            from langgraph.graph import Graph  # other versions
        except Exception:
            try:
                from langgraph import Graph
            except Exception:
                Graph = None

    # State: try common locations
    try:
        from langgraph.types import State
    except Exception:
        try:
            from langgraph.graph.state import State
        except Exception:
            State = None

    # MemorySaver (checkpoint)
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except Exception:
        MemorySaver = None

    if Graph is None or MemorySaver is None:
        missing = []
        if Graph is None:
            missing.append("Graph/StateGraph")
        if MemorySaver is None:
            missing.append("MemorySaver")
        raise ImportError(f"Missing langgraph components: {', '.join(missing)}")

except Exception as e:
    _graph_err = e
    # we'll raise later when Graph is required

# ------------------------------
# Config / paths / hyperparams
# ------------------------------
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
MAX_CONTEXT_CHARS = 15000
MAX_ITERATIONS = 3
SCORE_THRESHOLD = 9.0

# ------------------------------
# Simple state class
# ------------------------------
class IterationState(BaseModel):
    def __init__(self, task_id: str = "task_001"):
        self.task_id = task_id
        self.metadata: Dict[str, Any] = {"iteration": 0, "score": 0.0}
        self.story: str = ""
        self.summary: Dict[str, Any] = {}
        self.reviewer_feedback: Dict[str, Any] = {}
        self.human_feedback: Dict[str, Any] = {}

# ------------------------------
# Embedding and LLM wrappers
# ------------------------------
class Embedder:
    def __init__(self):
        self.use_openai = bool(os.getenv("OPENAI_API_KEY")) and openai is not None
        if self.use_openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model_name = "text-embedding-3-small"
        else:
            self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def embed_text(self, texts: List[str]) -> np.ndarray:
        if self.use_openai:
            embs = []
            for t in texts:
                resp = openai.Embedding.create(model=self.model_name, input=t)
                embs.append(np.array(resp["data"][0]["embedding"], dtype="float32"))
            return np.vstack(embs)
        else:
            embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            if embs.dtype != np.float32:
                embs = embs.astype("float32")
            return embs

class LLM:
    def __init__(self):
        self.use_openai = bool(os.getenv("OPENAI_API_KEY")) and openai is not None
        if self.use_openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def chat(self, system_prompt: str, user_prompt: str, temperature=0.2) -> str:
        if self.use_openai:
            resp = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=512,
            )
            return resp["choices"][0]["message"]["content"].strip()
        # fallback deterministic lightweight behavior:
        up = user_prompt.lower()
        if "summarize" in system_prompt.lower() or "summarize" in user_prompt.lower():
            toks = user_prompt.split()
            summary = " ".join(toks[:30])
            if len(toks) > 30:
                summary += "..."
            return summary
        if "score" in system_prompt.lower() or "review" in system_prompt.lower():
            import re
            words = re.findall(r"\b[0-9a-z']+\b", user_prompt)
            score = min(10, max(0, int(len(words) % 10)))
            return f"SCORE: {score}\nCOMMENT: fallback reviewer scored {score}/10"
        return user_prompt[:400]

# ------------------------------
# FAISS + metadata helpers
# ------------------------------
def init_or_load_faiss(index_path: str, meta_path: str, dim: int) -> Tuple[faiss.IndexIDMap, Dict[str, Any]]:
    if os.path.exists(index_path) and os.path.exists(meta_path):
        print("Loading FAISS index and metadata...")
        idx = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return idx, meta
    else:
        print("Creating new FAISS index and metadata...")
        index_flat = faiss.IndexFlatL2(dim)
        idx = faiss.IndexIDMap(index_flat)
        meta = {"next_id": 1, "by_id": {}, "mapping_task_iter": {}}
        return idx, meta

def save_faiss_and_meta(idx: faiss.IndexIDMap, meta: Dict[str, Any], index_path: str, meta_path: str):
    faiss.write_index(idx, index_path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"FAISS index saved to {index_path}; metadata saved to {meta_path}")

def upsert_iteration_to_faiss(idx: faiss.IndexIDMap, meta: Dict[str, Any],
                             embed: np.ndarray, task_id: str, thread_id: str,
                             iteration: int, score: float, summary_text: str,
                             reviewer_comment: str, created_at: str):
    key = f"{task_id}::iter::{iteration}"
    mapping = meta.get("mapping_task_iter", {})
    existing_id = mapping.get(key)
    if existing_id is not None:
        try:
            idx.remove_ids(np.array([existing_id], dtype=np.int64))
        except Exception:
            existing_id = None

    if existing_id is None:
        ext_id = meta["next_id"]
        meta["next_id"] = ext_id + 1
    else:
        ext_id = existing_id

    e = embed.astype("float32").reshape(1, -1)
    ids = np.array([ext_id], dtype=np.int64)
    idx.add_with_ids(e, ids)

    meta['mapping_task_iter'][key] = ext_id
    meta['by_id'][str(ext_id)] = {
        "faiss_id": int(ext_id),
        "task_id": task_id,
        "thread_id": thread_id,
        "iteration": iteration,
        "score": float(score),
        "created_at": created_at,
        "summary_text": summary_text,
        "reviewer_comment": reviewer_comment,
        "human_accepted": None
    }
    print(f"Upserted iteration {iteration} (task={task_id}) as id={ext_id}")

# ------------------------------
# Node implementations
# ------------------------------
embedder = Embedder()
llm = LLM()

if MemorySaver is None:
    raise RuntimeError(f"MemorySaver import failed earlier: {_graph_err}")
memory = MemorySaver()

def build_examples_from_metadata(meta: Dict[str, Any], full_text_limit_chars: int) -> str:
    items = []
    for ext_id_str, rec in meta.get("by_id", {}).items():
        snippet = f"ITER {rec['iteration']} (task {rec['task_id']}): {rec.get('summary_text','')[:400]} | score: {rec.get('score')}"
        items.append(snippet)
    big = "\n\n".join(items)
    if len(big) <= full_text_limit_chars:
        return big
    items2 = [f"ITER {rec['iteration']}: {rec.get('summary_text','')[:200]} | score:{rec.get('score')}" for rec in meta.get("by_id", {}).values()]
    small = "\n".join(items2)
    if len(small) <= full_text_limit_chars:
        return small
    items3 = [f"ITER {rec['iteration']}: {rec.get('summary_text','')[:100]} | score:{rec.get('score')}" for rec in meta.get("by_id", {}).values()]
    return "\n".join(items3[:2000])

def summarizer_node(state: IterationState):
    idx, meta = init_or_load_faiss(FAISS_INDEX_PATH, METADATA_PATH, EMBED_DIM)
    examples_blob = build_examples_from_metadata(meta, MAX_CONTEXT_CHARS)
    system_prompt = "You must summarize the story into <=30 words. Use examples below for guidance."
    user_prompt = f"Story:\n{state.story}\n\nExamples:\n{examples_blob}\n\nProduce a concise summary (<=30 words):"
    response = llm.chat(system_prompt, user_prompt, temperature=0.2)
    tokens = response.split()
    if len(tokens) > 30:
        response = " ".join(tokens[:30]) + "..."
    state.summary = {"text": response}
    print("[Summarizer] produced:", response)
    return state

def reviewer_node(state: IterationState):
    system_prompt = "You are a strict reviewer. Given story+summary, output SCORE: <0-10> and COMMENT: <one sentence>."
    user_prompt = f"Story:\n{state.story}\n\nSummary:\n{state.summary.get('text','')}\n\nReturn:\nSCORE: <number>\nCOMMENT: <one-sentence>"
    response = llm.chat(system_prompt, user_prompt, temperature=0.0)
    score = 0.0
    comment = ""
    for ln in [l.strip() for l in response.splitlines() if l.strip()]:
        if ln.lower().startswith("score"):
            try:
                score = float(ln.split(":",1)[1].strip())
            except Exception:
                pass
        elif ln.lower().startswith("comment"):
            comment = ln.split(":",1)[1].strip()
    if score == 0.0:
        import re
        m = re.search(r"\b([0-9](?:\.[0-9])?)\b", response)
        if m:
            try:
                score = float(m.group(1))
            except Exception:
                score = 0.0
    state.metadata["iteration"] = state.metadata.get("iteration", 0) + 1
    state.metadata["score"] = score
    state.reviewer_feedback = {"score": score, "comment": comment}
    print(f"[Reviewer] iter={state.metadata['iteration']} score={score} comment='{comment}'")
    return state

def human_feedback_node(state: IterationState, config: Dict[str, Any] = None):
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "T1")
    cps = memory.list(thread_id=thread_id)
    states = []
    for cp in cps:
        if isinstance(cp, dict) and "state" in cp:
            states.append(cp["state"])
        else:
            states.append(cp)
    found_iterations = {}
    for s in states:
        try:
            iter_no = s.metadata.get("iteration", 0)
            if iter_no and getattr(s, "summary", None):
                found_iterations[iter_no] = s
        except Exception:
            try:
                iter_no = s.get("metadata", {}).get("iteration", 0)
                if iter_no and "summary" in s:
                    found_iterations[iter_no] = s
            except Exception:
                pass
    if found_iterations:
        idx, meta = init_or_load_faiss(FAISS_INDEX_PATH, METADATA_PATH, EMBED_DIM)
        for iter_no in sorted(found_iterations.keys()):
            s = found_iterations[iter_no]
            summary_text = s.summary.get("text","") if hasattr(s,"summary") else (s.get("summary") or {}).get("text","")
            rv = s.reviewer_feedback if hasattr(s,"reviewer_feedback") else (s.get("reviewer_feedback") or {})
            score = rv.get("score",0.0)
            reviewer_comment = rv.get("comment", rv.get("note",""))
            story_text = s.story if hasattr(s,"story") else s.get("story","")
            iter_doc = f"Task:{s.task_id if hasattr(s,'task_id') else s.get('task_id','task')}\nIteration:{iter_no}\nSummary:{summary_text}\nScore:{score}\nReviewer:{reviewer_comment}\nStoryExcerpt:{story_text[:2000]}"
            emb = embedder.embed_text([iter_doc])[0]
            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            task_id = s.task_id if hasattr(s,"task_id") else s.get("task_id","task")
            upsert_iteration_to_faiss(idx, meta, emb, task_id, thread_id, iter_no, score, summary_text, reviewer_comment, created_at)
        save_faiss_and_meta(idx, meta, FAISS_INDEX_PATH, METADATA_PATH)
    else:
        print("[Human] No iterations found to persist from checkpoints.")
    print("\nFinal summary:\n", state.summary.get("text",""))
    while True:
        ans = input("Accept final summary? (y/n): ").strip().lower()
        if ans in ("y","n"):
            break
        print("type y or n")
    accepted = (ans == "y")
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH,"r",encoding="utf-8") as fh:
            meta = json.load(fh)
        for extid, rec in meta.get("by_id",{}).items():
            if rec.get("task_id")==state.task_id and rec.get("thread_id")==thread_id:
                rec["human_accepted"] = accepted
        with open(METADATA_PATH,"w",encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print(f"Updated human_accepted={accepted} for task {state.task_id}")
    state.human_feedback = {"accepted": accepted}
    return state

# ------------------------------
# Build graph & run
# ------------------------------
def build_graph():
    # if _graph_err:
    #     raise RuntimeError(f"langgraph import error: {_graph_err}")
    graph = StateGraph(IterationState)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("human_feedback", human_feedback_node)
    graph.add_edge("summarizer", "reviewer")
    def reviewer_decision(s):
        score = s.metadata.get("score",0)
        iter_n = s.metadata.get("iteration",0)
        if score>=SCORE_THRESHOLD or iter_n>=MAX_ITERATIONS:
            return "human_feedback"
        return "summarizer"
    graph.add_conditional_edges("reviewer", reviewer_decision)
    graph.set_entry_point("summarizer")
    try:
        graph.checkpointer = memory
    except Exception:
        try:
            graph.set_checkpointer(memory)
        except Exception:
            print("Warning: could not attach MemorySaver to graph via known methods.")
    return graph

def main():
    print("Agentic FAISS pipeline (per-iteration rows).")
    graph = build_graph()
    print("Paste the long story (end with empty line):")
    lines=[]
    while True:
        try:
            ln=input()
        except EOFError:
            break
        if ln.strip()=="":
            break
        lines.append(ln)
    story = "\n".join(lines).strip()
    if not story:
        print("No story input; exiting.")
        return
    state = IterationState(task_id="task_001")
    state.story = story
    config = {"configurable":{"thread_id":"T1"}}
    # try standard invoke signature
    try:
        final_state = graph.invoke(state, config=config)
    except TypeError:
        try:
            final_state = graph.invoke(state)
        except Exception as e:
            raise
    print("Done. Human feedback:", getattr(final_state,"human_feedback",{}))

if __name__=="__main__":
    main()
