import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from langgraph.graph.state import StateT
from pydantic import BaseModel, Field


from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import uuid
import re
import re
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_INDEX_PATH = "vectors.npy"          # ✅ match the actual file being saved
METADATA_DB = "vector_metadata.pkl"


# -------------------- STATE DEFINITIONS --------------------
class IterationSnapshot(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    metadata: Dict[str, Any] = {
        "task_name": "",
        "iteration": 0,
        "score": 0.0
    }

    generator_inputs: Dict[str, Any] = {
        "user_task": ""
    }

    generator_output: Dict[str, Any] = {
        "final_ans": "",
        "reasoning": "",
        "bulletlist": []
    }

    reflector_output: Dict[str, Any] = {
        "reasoning": "",
        "error_identification": [],
        "rootcauseanalysis": "",
        "correctapproach": "",
        "keyinsights": [],
        "bullet_tags": {}
    }

    curator_output: Dict[str, Any] = {
        "action": [],
        "content": "",
        'title':"",
        "score": 0.0,
        "bullet_tags": {}
    }

    consolidation_output: Dict[str, Any] = {
        "playbook_update_boolean": False
    }

    human_feedback_output: Dict[str, Any] = {
        "human_approval_status": None,
        "human_feedback": None
    }


class LangGraphState(BaseModel):
    iterations: List[IterationSnapshot] = []


# -------------------- LLM HELPER --------------------
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    return llm(messages).content



# -----------------------------
# Simple Vector DB
# -----------------------------
class SimpleVectorDB:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = None
        self.metadb = {}
        self._load()

    def _load(self):
        if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(METADATA_DB):
            xb = np.load("vectors.npy")
            d = xb.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(xb)
            with open(METADATA_DB, "rb") as f:
                self.metadb = pickle.load(f)
        else:
            self.index = None
            self.metadb = {}

    def _save(self, xb: Optional[np.ndarray] = None):
        if xb is not None:
            np.save("vectors.npy", xb)
        with open(METADATA_DB, "wb") as f:
            pickle.dump(self.metadb, f)

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, convert_to_numpy=True))

    def upsert(self, docs: List[Dict[str, Any]]):
        texts = [d["text"] for d in docs]
        xs = self.embed(texts)
        self._ensure_index(xs.shape[1])
        start_idx = self.index.ntotal if self.index is not None else 0
        self.index.add(xs)
        for i, d in enumerate(docs):
            key = start_idx + i
            self.metadb[key] = d
        self._save(self._get_all_vectors())

    def _get_all_vectors(self):
        if len(self.metadb) > 0:
            texts = [self.metadb[k]["text"] for k in sorted(self.metadb.keys())]
            return self.embed(texts)
        else:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))

    def query(self, query_text: str, top_k: int = 3):
        xq = self.embed([query_text])
        if self.index is None or self.index.ntotal == 0:
            return []
        D, I = self.index.search(xq, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append((self.metadb[idx], float(dist)))
        return results

vdb = SimpleVectorDB()


def fetch_all_from_simple_vdb(vdb: SimpleVectorDB) -> List[Dict[str, Any]]:
    """
    Return all stored records from the simple vdb as a list of dicts, sorted by stored key.
    Each dict should contain fields like: text, metadata (which itself can include task_id, metadata keys).
    """
    records = []
    if not hasattr(vdb, "metadb"):
        return records

    for k in sorted(vdb.metadb.keys()):
        records.append(vdb.metadb[k])
    return records

# will be used for vectordb insert
def build_task_memory_text(task_state: LangGraphState) -> str:
    """
    Build a single combined text for the entire execution of a task (all iterations),
    suitable to store as one row in the vector DB.
    """
    parts = []
    for it in task_state.iterations:
        md = it.metadata or {}
        parts.append(f"Task ID: {getattr(it, 'task_id', md.get('task_id',''))}")
        parts.append(f"Iteration: {md.get('iteration')}")
        parts.append(f"Task name: {md.get('task_name')}")
        parts.append(f"Score: {md.get('score')}")
        parts.append(f"User task: {it.generator_inputs.get('user_task')}")
        parts.append("Final answer:")
        parts.append(str(it.generator_output.get("final_ans", "")))
        parts.append("Reasoning:")
        parts.append(str(it.generator_output.get("reasoning", "")))
        bullets = it.generator_output.get("bulletlist", []) or []
        if bullets:
            parts.append("Bullets: " + "; ".join(bullets))
        # reflector + curator + human feedback
        parts.append("Reflector reasoning:")
        parts.append(str(it.reflector_output.get("reasoning", "")))
        parts.append("Reflector errors:")
        parts.append(str(it.reflector_output.get("error_identification", [])))
        parts.append("Curator content:")
        parts.append(str(it.curator_output.get("content", "")))
        parts.append("Curator score:")
        parts.append(str(it.curator_output.get("score", "")))
        parts.append("Human approval:")
        parts.append(str(it.human_feedback_output.get("human_approval_status", "")))
        parts.append("-" * 80)
    return "\n".join(parts)


def _format_all_vdb_records_for_prompt(records: List[Dict[str, Any]]) -> str:
    """
    Convert all records (each row contains a full task-execution text and metadata)
    into a readable memory summary that we inject into the system prompt.
    """
    if not records:
        return "No prior memory found."

    lines = []
    for rec in records:
        # rec expected to have keys like: 'text' (full execution text) and 'metadata' or other fields
        meta = rec.get("metadata", {}) if isinstance(rec.get("metadata", {}), dict) else {}
        task_id = meta.get("task_id", rec.get("task_id", ""))
        task_name = meta.get("task_name", "")
        total_iters = meta.get("total_iterations", meta.get("iterations") or "")
        score = meta.get("score", "")
        human_status = meta.get("human_feedback_output", {}).get("human_approval_status") if meta.get("human_feedback_output") else rec.get("human_feedback_output", {}).get("human_approval_status", None)

        lines.append(f"- task_id: {task_id} | name: {task_name} | iters: {total_iters} | score: {score} | human_approved: {human_status}")
        # prefer stored textual execution under 'text' or Document-style 'page_content'
        text = rec.get("text") or rec.get("page_content") or rec.get("raw_text") or ""
        if text:
            # show only a preview (first ~1000 chars) to keep prompt readable; model still receives full memory if you want
            preview = text if len(text) <= 2000 else (text[:2000] + "...(truncated)")
            lines.append(preview)
        else:
            # fallback to any metadata summary fields if present
            if meta:
                lines.append("metadata: " + json.dumps(meta, default=str))
        lines.append("")  # blank line between records

    return "\n".join(lines)


# Modified generator node that uses SimpleVectorDB (reads entire DB and upserts the current task)

# --- Helpers to fetch playbook rows from SimpleVectorDB.metadb ---
def fetch_full_playbook_from_vdb(vdb: SimpleVectorDB) -> List[Dict[str, Any]]:
    """
    Return a list of playbook records (full dicts) from vdb.metadb.
    Recognizes records whose metadata.type == "playbook" OR where record has 'bullet_id' starting with 'B-'.
    """
    recs = []
    if not hasattr(vdb, "metadb"):
        return recs

    for k in sorted(vdb.metadb.keys()):
        row = vdb.metadb[k]
        # row may contain metadata inside row.get("metadata") or have top-level keys
        meta = row.get("metadata", {}) if isinstance(row, dict) else {}
        bullet_id = meta.get("bullet_id") or row.get("bullet_id") or (meta.get("id") if isinstance(meta.get("id"), str) and meta.get("id").startswith("B-") else None)
        row_type = (meta.get("type") or row.get("type") or "").lower()
        if row_type == "playbook" or (isinstance(bullet_id, str) and bullet_id.startswith("B-")):
            # ensure helpful_content/harmful_content keys exist
            if "metadata" in row and isinstance(row["metadata"], dict):
                row_meta = row["metadata"]
            else:
                row_meta = meta
            # normalize fields into a canonical shape for downstream use
            rec = {
                "key": k,
                "bullet_id": row_meta.get("bullet_id") or row.get("bullet_id") or row_meta.get("id") or "",
                "title": row_meta.get("title") or row.get("title") or "",
                "content": row_meta.get("content") or row.get("content") or row.get("text") or "",
                "helpful_content": int(row_meta.get("helpful_content", 0)),
                "harmful_content": int(row_meta.get("harmful_content", 0)),
                "raw": row
            }
            recs.append(rec)
    return recs


def format_playbook_for_prompt(playbook_rows: List[Dict[str, Any]]) -> str:
    """
    Convert playbook rows into a human-readable block the LLM can parse and decide from.
    Each entry will be labeled with its bullet id.
    """
    if not playbook_rows:
        return "No playbook bullets found."

    parts = []
    for p in playbook_rows:
        parts.append(f"{p['bullet_id']} | {p['title']}\n{p['content']}\nhelpful={p['helpful_content']} harmful={p['harmful_content']}")
        parts.append("")  # blank line between bullets
    return "\n".join(parts)


# --- Main generator that uses playbook ---
def generator_node_with_playbook(
    state: LangGraphState,
    vdb: SimpleVectorDB,
    playbook_override: Optional[List[Dict[str, Any]]] = None
) -> LangGraphState:
    """
    - fetch full DB for memory (as before)
    - fetch the entire playbook (from vdb.metadb)
    - inject playbook text into system prompt and require LLM to return BULLET_IDS: in addition
      to FINAL_ANS:, REASONING:, BULLETS:
    - parse BULLET_IDS, update helpful_content counts in vdb.metadb
    """
    task_id = str(uuid.uuid4())
    iteration_num = len(state.iterations) + 1
    user_task = state.iterations[-1].generator_inputs.get("user_task") if state.iterations else "Define task"

    # fetch entire task memory (all DB rows)
    all_records = fetch_all_from_simple_vdb(vdb)
    memory_summary = _format_all_vdb_records_for_prompt(all_records)

    # fetch playbook rows (from vdb.metadb)
    if playbook_override is not None:
        playbook_rows = playbook_override
    else:
        playbook_rows = fetch_full_playbook_from_vdb(vdb)

    playbook_text = format_playbook_for_prompt(playbook_rows)

    # System prompt: require structured response including BULLET_IDS
    system_prompt = (
    "You are the Generator node. Use the full memory for self learning and the entire playbook bullets (using title and content) below to generate the answer. "
    "Return EXACTLY the following labeled sections (uppercase labels). Do not add other text.\n\n"
    "FINAL_ANS:\n"
    "  - Provide Excel formula(s) with column names or cell references. Ready to paste.\n\n"
    "REASONING:\n"
    "  - Explain why the formula(s) work and any assumptions/edge-cases.\n\n"
    "BULLETS:\n"
    "  - Use 'title' and 'content' columns data to enhance the output, where bullet id column is the reference to the record in playbook.\n\n"
    "For example, from the sample record of the playbook - "
    "{\"bullet_id\": \"B-104\", \"Title\": \"IQR Calculation\", \"Content\": \"Always sort the data before calculating percentiles\"}.\n\n"
    "BULLET_IDS:\n"
    "  - A comma-separated list of playbook bullet ids (e.g. B-101, B-107) that you USED from the playbook above. "
    "Only include ids that you actually used to decide the formula or checks. If none, return NONE.\n\n"
    "PLAYBOOK (choose from these bullets):\n"
    f"{playbook_text}\n\n"
    "FULL MEMORY:\n"
    f"{memory_summary}\n\n"
    "Now produce the sections for task:\n"
)

    user_prompt = f"Task: {user_task}"

    raw_output = call_llm(system_prompt, user_prompt)

    # --- parse sections: FINAL_ANS, REASONING, BULLETS, BULLET_IDS ---
    normalized = raw_output.replace("\r\n", "\n")

    def extract_section_by_labels(text: str, start_label: str, next_labels: List[str]) -> str:
        """
        Extract text of start_label until next label occurrence among next_labels or end of text.
        """
        # find start
        s_idx = text.find(start_label)
        if s_idx < 0:
            return ""
        s_idx += len(start_label)
        # find nearest next label
        next_positions = [text.find(lbl, s_idx) for lbl in next_labels if text.find(lbl, s_idx) >= 0]
        if next_positions:
            end_idx = min(next_positions)
            return text[s_idx:end_idx].strip()
        else:
            return text[s_idx:].strip()

    labels = ["FINAL_ANS:", "REASONING:", "BULLETS:", "BULLET_IDS:"]
    final_ans_text = extract_section_by_labels(normalized, "FINAL_ANS:", labels[1:])
    reasoning_text = extract_section_by_labels(normalized, "REASONING:", labels[2:])
    bullets_text = extract_section_by_labels(normalized, "BULLETS:", ["BULLET_IDS:","FINAL_ANS:","REASONING:"])
    bullet_ids_text = extract_section_by_labels(normalized, "BULLET_IDS:", [])

    # parse bullet ids: accept comma separated or newline list or "NONE"
    parsed_bullet_ids = []
    if bullet_ids_text:
        txt = bullet_ids_text.strip()
        if txt.upper() == "NONE":
            parsed_bullet_ids = []
        else:
            # split by commas or whitespace/newlines
            parts = re.split(r"[,;\n]+", txt)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # keep only items that look like B-XXX
                if re.match(r"^B-\d+", p, re.IGNORECASE):
                    parsed_bullet_ids.append(p.upper())
                else:
                    # allow titles like "B-101 (IQR calc)" -> extract B-101
                    m = re.search(r"(B-\d+)", p, re.IGNORECASE)
                    if m:
                        parsed_bullet_ids.append(m.group(1).upper())

    # convert bullets_text to list
    # bullet_list = []
    # for ln in bullets_text.splitlines():
    #     ln = ln.strip()
    #     if not ln:
    #         continue
    #     if ln.startswith("-"):
    #         bullet_list.append(ln.lstrip("- ").strip())
    #     else:
    #         bullet_list.append(ln)

    # create iteration and write to state
    print("this is generator final ans:", final_ans_text)
    print("this is generator's reasoning:", reasoning_text)
    iteration = IterationSnapshot(
        task_id=task_id,
        metadata={"task_name": user_task, "iteration": iteration_num, "score": 0.0},
        generator_inputs={"user_task": user_task},
        generator_output={
            "final_ans": final_ans_text,
            "reasoning": reasoning_text,
            "bulletlist": parsed_bullet_ids,
        }
    )
    state.iterations.append(iteration)

    # --- update playbook counters in vdb.metadb: increment helpful_content for chosen bullets ---
    if parsed_bullet_ids:
        # map bullet_id -> record key(s) (there may be multiple stored rows; match first)
        for used_bid in parsed_bullet_ids:
            # find the playbook entry in vdb.metadb
            for k in list(vdb.metadb.keys()):
                row = vdb.metadb[k]
                meta = row.get("metadata", {}) if isinstance(row, dict) else {}
                candidate_id = meta.get("bullet_id") or row.get("bullet_id") or meta.get("id") or row.get("id")
                if candidate_id and str(candidate_id).upper() == used_bid:
                    # increase helpful_content in metadata
                    if "metadata" not in row or not isinstance(row["metadata"], dict):
                        row_meta = meta if isinstance(meta, dict) else {}
                        row["metadata"] = row_meta
                    else:
                        row_meta = row["metadata"]
                    row_meta["helpful_content"] = int(row_meta.get("helpful_content", 0)) + 1
                    # also update our canonical playbook_rows structure if present (not required)
                    # write back into vdb.metadb
                    vdb.metadb[k] = row
                    break

        # persist metadb (SimpleVectorDB._save writes metadb to METADATA_DB)
        try:
            vdb._save()  # uses internal save; ok to call here
        except Exception as e:
            print("Warning: failed to save updated playbook helpful counts:", e)

    # Optionally, you could also increment harmful_content for bullets the model explicitly said were confusing.
    # That would require the model to return a separate list of harmful ids, or you infer based on not-used bullets — but
    # inferring harmful by not-used could be noisy, so I avoided that.

    # Finally, return updated state
    # (you can also store the combined text for the current task as before if you want)
    return state

# -------------------- NODES IMPLEMENTATION --------------------
# def generator_node(state: LangGraphState) -> LangGraphState:
#     task_id = str(uuid.uuid4())
#     iteration_num = len(state.iterations) + 1

#     user_task = state.iterations[-1].generator_inputs.get("user_task") if state.iterations else "Define task"

#     system_prompt = (
#         "You are the Generator node in the ACE architecture. "
#         "Produce high-quality output for the given task with reasoning and bullet points."
#     )
#     user_prompt = f"Task: {user_task}"

#     output = call_llm(system_prompt, user_prompt)

#     # Split reasoning and bullets heuristically
#     reasoning = output
#     bullets = [line.strip("- ") for line in output.split("\n") if line.strip().startswith("-")]

#     iteration = IterationSnapshot(
#         task_id=task_id,
#         metadata={"task_name": user_task, "iteration": iteration_num, "score": 0.0},
#         generator_inputs={"user_task": user_task},
#         generator_output={"final_ans": output, "reasoning": reasoning, "bulletlist": bullets}
#     )

#     state.iterations.append(iteration)
#     return state


import json
from typing import List, Dict, Any

def reflector_node(state: LangGraphState) -> LangGraphState:
    iteration = state.iterations[-1]

    # Present-context (what we must analyze now) — use generator's output as primary context
    gen_final = iteration.generator_output.get("final_ans", "")
    gen_reasoning = iteration.generator_output.get("reasoning", "")
    gen_bullets = iteration.generator_output.get("bulletlist", []) or []
    user_task = iteration.generator_inputs.get("user_task", "")

    # ------------------------------------------------------------------
    # Build full history from vectordb (entire metadb), normalized to fields required
    # ------------------------------------------------------------------
    history_items: List[Dict[str, Any]] = []
    try:
        if vdb is not None and hasattr(vdb, "metadb"):
            all_rows = fetch_all_from_simple_vdb(vdb)  # returns list of dicts (vdb.metadb entries) sorted by key
        else:
            all_rows = []
    except Exception:
        all_rows = []

    # Normalize each record to the required schema
    for rec in all_rows:
        # rec might be the saved doc dict; metadata often under rec["metadata"]
        meta = rec.get("metadata", {}) if isinstance(rec, dict) else {}
        # safe getters
        def get_from_rec(path_list, default=""):
            cur = rec
            for p in path_list:
                if not isinstance(cur, dict):
                    return default
                cur = cur.get(p, None)
                if cur is None:
                    return default
            return cur

        history_items.append({
            "task_id": meta.get("task_id") or rec.get("task_id") or "",
            "metadata.task_name": meta.get("task_name") or rec.get("task_name") or "",
            "generators_input_user_task": get_from_rec(["generator_inputs", "user_task"]) or meta.get("user_task") or rec.get("user_task") or "",
            "generators_output_final_ans": get_from_rec(["generator_output", "final_ans"]) or rec.get("final_ans", ""),
            "generators_output_reasoning": get_from_rec(["generator_output", "reasoning"]) or rec.get("reasoning", ""),
            "generators_output_bulletlist": get_from_rec(["generator_output", "bulletlist"]) or rec.get("bulletlist", []) or [],
            "human_feedback_output_human_approval_status": get_from_rec(["human_feedback_output", "human_approval_status"]) or (meta.get("human_feedback_output") or {}).get("human_approval_status") or rec.get("human_feedback_output", {}).get("human_approval_status"),
            "reflector_output_reasoning": get_from_rec(["reflector_output", "reasoning"]) or (rec.get("reflector_output") or {}).get("reasoning"),
            "reflector_output_error_identification": get_from_rec(["reflector_output", "error_identification"]) or (rec.get("reflector_output") or {}).get("error_identification"),
            "reflector_output_keyinsights": get_from_rec(["reflector_output", "keyinsights"]) or (rec.get("reflector_output") or {}).get("keyinsights"),
            "reflector_output_correctapproach": get_from_rec(["reflector_output", "correctapproach"]) or (rec.get("reflector_output") or {}).get("correctapproach"),
        })

    # ------------------------------------------------------------------
    # Build prompt: include compact history (full set may be large; we include previews up to N characters each)
    # ------------------------------------------------------------------
    def preview_hist_item(it: Dict[str, Any], max_chars: int = 800) -> str:
        preview = {
            "task_id": it.get("task_id"),
            "metadata.task_name": it.get("metadata.task_name"),
            "generators_input_user_task": (it.get("generators_input_user_task") or "")[:max_chars],
            "generators_output_final_ans": (it.get("generators_output_final_ans") or "")[:max_chars],
            "generators_output_reasoning": (it.get("generators_output_reasoning") or "")[:max_chars],
            "generators_output_bulletlist": it.get("generators_output_bulletlist", []),
            "human_feedback": it.get("human_feedback_output_human_approval_status"),
            "reflector_keyinsights": it.get("reflector_output_keyinsights")
        }
        return json.dumps(preview, ensure_ascii=False)

    # join entire history previews (if extremely large, you may want to truncate or paginate)
    history_block = "\n".join(preview_hist_item(h) for h in history_items)

    if not history_block:
        history_block = "No prior history available."


    schema_instructions = (
        "Return JSON with the following top-level keys:\n"
        "{\n"
        #  '  "task_id": string|null,\n'
        # '  "metadata_task_name": string|null,\n'
        # '  "generators_input_user_task": string|null,\n'
        # '  "generators_output_final_ans": string,\n'
        # '  "generators_output_reasoning": string,\n'
        # '  "generators_output_bulletlist": [string],\n'
        '  "reflector_output": {\n'
        '      "reasoning": string,\n'
        '      "error_identification": [string],\n'
        '      "rootcauseanalysis": string,\n'
        '      "correctapproach": string,\n'
        '      "keyinsights": [string],\n'
        '      "bullet_tags": { "<bullet_id_or_text>": "helpful" | "harmful", ... }\n'
        '  }\n'
        '}\n'
    )
    # System + user prompt requesting strict JSON with required fields
    system_prompt = (
    "You are the Reflector node in the ACE architecture. Learn from the execution history below and analyze"
    " the present generator output. Return ONLY a valid JSON object (no extra commentary) with the exact schema requested.\n\n"
    f"{schema_instructions}\n\n"
    "HISTORY (entire DB previews):\n"
    f"{history_block}"
)


    user_prompt = (
    "Present generator output (primary context):\n"
    f"FINAL_ANS:\n{gen_final}\n\n"
    f"REASONING:\n{gen_reasoning}\n\n"
    f"BULLETLIST:\n{json.dumps(gen_bullets, ensure_ascii=False)}\n\n"
    f"TASK INPUT:\n{user_task}\n\n"
    "Analyze and return JSON as instructed in the system message."
)

    

    # ------------------------------------------------------------------
    # Call LLM and parse JSON robustly
    # ------------------------------------------------------------------
    raw_response = call_llm(system_prompt, user_prompt)

    parsed = None
    try:
        parsed = json.loads(raw_response)
    except Exception:
        # try to extract JSON blob
        try:
            start = raw_response.index("{")
            end = raw_response.rindex("}") + 1
            parsed = json.loads(raw_response[start:end])
        except Exception:
            parsed = None

    print("yeh reflector pe print ho rha:", parsed)

    # ------------------------------------------------------------------
    # If parsing fails, fall back to putting raw response into reasoning field
    # ------------------------------------------------------------------
    if not isinstance(parsed, dict):
        iteration.reflector_output.update({
            "reasoning": raw_response,
            "error_identification": ["analysis_not_parsed_from_llm"],
            "rootcauseanalysis": "",
            "correctapproach": "",
            "keyinsights": [],
            "bullet_tags": {b: "harmful" for b in gen_bullets}  # conservative fallback
        })
        return state

    # ------------------------------------------------------------------
    # Normalize parsed result and update iteration.reflector_output
    # ------------------------------------------------------------------
    reflector_block = parsed.get("reflector_output", {})

    iteration.reflector_output.update({
        "reasoning": reflector_block.get("reasoning", "") or parsed.get("reflector_output", {}).get("reasoning", ""),
        "error_identification": reflector_block.get("error_identification", []) or parsed.get("error_identification", []),
        "rootcauseanalysis": reflector_block.get("rootcauseanalysis", "") or parsed.get("rootcauseanalysis", ""),
        "correctapproach": reflector_block.get("correctapproach", "") or parsed.get("correctapproach", ""),
        "keyinsights": reflector_block.get("keyinsights", []) or parsed.get("keyinsights", []),
        "bullet_tags": reflector_block.get("bullet_tags", {}) or parsed.get("bullet_tags", {})
    })

    # Ensure all bullets from the present generator are present in bullet_tags (conservative default = "harmful")
    btags = iteration.reflector_output.get("bullet_tags", {})
    for b in gen_bullets:
        if b not in btags:
            btags[b] = "harmful"
    iteration.reflector_output["bullet_tags"] = btags

    # Optionally store the top-level metadata fields (task_id, metadata_task_name, generators_input_user_task)
    # on the reflector_output as convenience (not required, but requested fields)
    # We try to source them from parsed JSON first, otherwise from present iteration.
    iteration.reflector_output.setdefault("task_id", parsed.get("task_id") or iteration.task_id if hasattr(iteration, "task_id") else parsed.get("task_id"))
    iteration.reflector_output.setdefault("metadata_task_name", parsed.get("metadata_task_name") or iteration.metadata.get("task_name"))
    iteration.reflector_output.setdefault("generators_input_user_task", parsed.get("generators_input_user_task") or user_task)

    return state



def curator_node(state: LangGraphState) -> LangGraphState:
    """
    Curator:
      - Receives present context (generator final_ans + reflector outputs).
      - Receives entire playbook (from vdb).
      - Decides whether to 'add', 'update' or 'ignore' the candidate playbook bullet.
      - Returns strict JSON with curator_output.action in {'add','update','ignore'} plus content/title/optional bullet_id and a numeric score.
    """
    iteration = state.iterations[-1]

    # Present context from reflector + generator final_ans
    present_task_name = iteration.metadata.get("task_name", "")
    gen_final_ans = iteration.generator_output.get("final_ans", "")
    refl_reasoning = iteration.reflector_output.get("reasoning", "")
    refl_errors = iteration.reflector_output.get("error_identification", [])
    refl_rootcause = iteration.reflector_output.get("rootcauseanalysis", "")
    refl_correct = iteration.reflector_output.get("correctapproach", "")
    refl_keyinsights = iteration.reflector_output.get("keyinsights", [])

    # Fetch entire playbook (canonical rows)
    try:
        playbook_rows = fetch_full_playbook_from_vdb(vdb) if (vdb is not None and hasattr(vdb, "metadb")) else []
    except Exception:
        playbook_rows = []

    playbook_text = format_playbook_for_prompt(playbook_rows) if playbook_rows else "No playbook bullets found."

    # Schema: require exact keys and an action among add/update/ignore
    schema_instructions = (
        "Return ONLY a valid JSON object (no extra commentary) using the exact structure below.\n\n"
        "{\n"
        '  "metadata_task_name": string|null,\n'
        '  "generator_final_ans": string,\n'
        '  "curator_output": {\n'
        '      "action": "add" | "update" | "ignore",\n'
        '      "title": string|null,          # required when action == "add" or when suggesting title for update\n'
        '      "content": string|null,        # required when action == "add" or when suggesting refined content for update\n'
        '      "bullet_id": string|null,      # required when action == "update" (the existing playbook entry id, e.g. B-101)\n'
        '      "score": number                # 1-10, judge of the generator output usefulness\n'
        '  }\n'
        '}\n'
    )

    system_prompt = (
        "You are the Curator node in the ACE architecture. Your job is to examine the present generator output and the Reflector's analysis, "
        "compare the candidate learning bullet against the ENTIRE playbook below, and decide whether to ADD a new playbook bullet, UPDATE an existing one, or IGNORE it.\n\n"
        f"{schema_instructions}\n\n"
        "PLAYBOOK (entire):\n"
        f"{playbook_text}\n\n"
        "Important:\n"
        "- If you choose 'update', include the exact 'bullet_id' of the existing entry you want to update.\n"
        "- If you choose 'add', provide a concise 'title' (<= 12 words) and an actionable 'content' suitable for the playbook.\n"
        "- If you choose 'ignore', set 'title' and 'content' to null.\n"
        "- Score is your rating (1-10) of the GENERATOR'S FINAL_ANS usefulness/correctness for the task.\n"
        "- Return the JSON object and nothing else."
    )

    user_prompt = (
        "Present context (from Reflector + Generator):\n"
        f"Task name: {present_task_name}\n\n"
        "Generator FINAL_ANS:\n"
        f"{gen_final_ans}\n\n"
        "Reflector reasoning:\n"
        f"{refl_reasoning}\n\n"
        "Reflector error_identification:\n"
        f"{json.dumps(refl_errors, ensure_ascii=False)}\n\n"
        "Reflector rootcauseanalysis:\n"
        f"{refl_rootcause}\n\n"
        "Reflector correctapproach:\n"
        f"{refl_correct}\n\n"
        "Reflector keyinsights (candidate learning points):\n"
        f"{json.dumps(refl_keyinsights, ensure_ascii=False)}\n\n"
        "Using the full playbook shown in the system message, decide whether the candidate insight should be added, used to refine an existing bullet (update), or ignored.\n"
        "Return the JSON described in the system message."
    )

    raw = call_llm(system_prompt, user_prompt)

    # Robust JSON parsing
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
        except Exception:
            parsed = None

    # Debug print (optional)
    print("CURATOR parsed:", parsed)

    # Fallback if parsing failed
    if not isinstance(parsed, dict):
        # extract score if possible
        match = re.search(r"score\s*[:\-]?\s*(\d+(\.\d+)?)", raw, re.IGNORECASE)
        score = float(match.group(1)) if match else 7.0
        # fallback decision heuristic: if score >=9 -> add, 6-8 -> update, else ignore
        if score >= 9:
            action = ["add"]
            title = present_task_name or None
            content = refl_keyinsights[0] if refl_keyinsights else (refl_reasoning[:200] or None)
            bullet_id = None
        elif score >= 6:
            action = ["update"]
            title = present_task_name or None
            content = refl_keyinsights[0] if refl_keyinsights else (refl_reasoning[:200] or None)
            # best-effort: choose first playbook match by title if exists
            bullet_id = None
            for r in playbook_rows:
                if (r.get("title") or "").strip().lower() == (title or "").strip().lower():
                    bullet_id = r.get("bullet_id") or (r.get("raw") or {}).get("metadata", {}).get("bullet_id")
                    break
        else:
            action = ["ignore"]
            title = None
            content = None
            bullet_id = None

        iteration.curator_output.update({
            "action": action,
            "content": content,
            "title": title,
            "bullet_id": bullet_id,
            "score": score
        })
        iteration.metadata["score"] = score
        return state

    # Parsed OK — normalize fields
    curator_block = parsed.get("curator_output", {})
    action_raw = curator_block.get("action")
    # accept either a single string or list — normalize to list for compatibility
    if isinstance(action_raw, list):
        action = [str(a).strip().lower() for a in action_raw]
    else:
        action = [str(action_raw).strip().lower()] if action_raw is not None else []

    # Ensure action value is one of expected
    allowed = {"add", "update", "ignore"}
    final_action = None
    for a in action:
        if a in allowed:
            final_action = a
            break
    if final_action is None:
        # fallback to 'ignore' if model returns unexpected action
        final_action = "ignore"

    content = curator_block.get("content") if curator_block.get("content") not in (None, "") else None
    title = curator_block.get("title") if curator_block.get("title") not in (None, "") else None
    bullet_id = curator_block.get("bullet_id") if curator_block.get("bullet_id") not in (None, "") else None

    score = curator_block.get("score", None)
    if score is None:
        # try top-level score or regex fallback
        top_score = parsed.get("score")
        if isinstance(top_score, (int, float)):
            score = float(top_score)
        else:
            m = re.search(r"score\s*[:\-]?\s*(\d+(\.\d+)?)", json.dumps(parsed), re.IGNORECASE)
            score = float(m.group(1)) if m else 7.0

    # Normalize types
    try:
        score = float(score)
    except Exception:
        score = 7.0

    # Update iteration.curator_output and metadata
    iteration.curator_output.update({
        "action": [final_action],
        "content": content,
        "title": title,
        "bullet_id": bullet_id,
        "score": score
    })
    iteration.metadata["score"] = score

    return state



MAX_ITERATIONS = 3

def consolidator_node(state: LangGraphState) -> LangGraphState:
    """
    Consolidator: strictly follow the curator's instruction.
      - action in curator_output.action should be one of: ["add"], ["update"], ["ignore"]
      - when action == "update": require curator_output.bullet_id (e.g. "B-101"); update that record if found
      - when action == "add": create a new playbook record with a new bullet_id
      - when action == "ignore": do nothing
    Sets iteration.consolidation_output["playbook_update_boolean"] = True only when create/update succeeded.
    """
    iteration = state.iterations[-1]
    playbook_updated = False

    # Read curator outputs
    curator_action = iteration.curator_output.get("action", []) or []
    if isinstance(curator_action, str):
        curator_action = [curator_action]
    curator_action_lc = [str(a).strip().lower() for a in curator_action]

    # normalize single action if provided
    action = curator_action_lc[0] if curator_action_lc else "ignore"

    curator_title = (iteration.curator_output.get("title") or "").strip()
    curator_content = (iteration.curator_output.get("content") or "").strip()
    curator_bullet_tags = iteration.curator_output.get("bullet_tags", {}) or {}
    curator_bullet_id = iteration.curator_output.get("bullet_id") or None

    # quick helper to ensure metadata counters exist
    def ensure_counters(meta: dict):
        meta["helpful_count"] = int(meta.get("helpful_count", meta.get("helpful_content", 0) or 0))
        meta["harmful_count"] = int(meta.get("harmful_count", meta.get("harmful_content", 0) or 0))

    # helper to apply tags increments
    def apply_tags(meta: dict, tags: Dict[str, Any]):
        ensure_counters(meta)
        for _k, v in (tags or {}).items():
            if isinstance(v, str) and v.lower() == "helpful":
                meta["helpful_count"] += 1
            elif isinstance(v, str) and v.lower() == "harmful":
                meta["harmful_count"] += 1

    # Load playbook rows (canonicalized)
    try:
        playbook_rows = fetch_full_playbook_from_vdb(vdb) if (vdb is not None and hasattr(vdb, "metadb")) else []
    except Exception:
        playbook_rows = []

    # ACTION: ignore -> nothing to do
    if action == "ignore":
        iteration.consolidation_output["playbook_update_boolean"] = False
        return state

    # ACTION: update -> require bullet_id
    if action == "update":
        if not curator_bullet_id:
            # nothing to update (curator asked update but didn't supply id)
            iteration.consolidation_output["playbook_update_boolean"] = False
            return state

        # find the matching playbook entry by bullet_id
        matched = None
        matched_key = None
        for row in playbook_rows:
            bid = (row.get("bullet_id") or "") or ((row.get("raw") or {}).get("metadata") or {}).get("bullet_id") or ""
            if isinstance(bid, str) and bid.strip().upper() == str(curator_bullet_id).strip().upper():
                matched = row
                matched_key = row["key"]
                break

        if matched is None:
            # no matching bullet_id found -> do nothing (consolidator follows curator exactly)
            iteration.consolidation_output["playbook_update_boolean"] = False
            return state

        # update the found record
        stored = matched.get("raw", {}) or vdb.metadb.get(matched_key, {}) or {}
        meta = stored.setdefault("metadata", {})

        # Update title/content if curator provided them (follow curator's instruction)
        if curator_title:
            meta["title"] = curator_title
        if curator_content:
            meta["content"] = curator_content

        # apply tag increments
        apply_tags(meta, curator_bullet_tags)

        # persist
        vdb.metadb[matched_key] = stored
        try:
            vdb._save()
            playbook_updated = True
        except Exception as e:
            print("Warning: failed to save updated playbook entry:", e)
            playbook_updated = False

        iteration.consolidation_output["playbook_update_boolean"] = bool(playbook_updated)
        return state

    # ACTION: add -> always create a new playbook entry (even if similar exists; curator decided)
    if action == "add":
        # generate next bullet id by scanning existing ids
        max_bnum = 100
        for k, row in vdb.metadb.items():
            try:
                meta = (row.get("metadata") if isinstance(row, dict) else {}) or {}
                bid = meta.get("bullet_id") or row.get("bullet_id") or meta.get("id") or row.get("id")
                if bid and isinstance(bid, str):
                    m = re.search(r"B-?0*?(\d+)$", bid, re.IGNORECASE)
                    if m:
                        n = int(m.group(1))
                        if n > max_bnum:
                            max_bnum = n
            except Exception:
                continue
        new_num = max_bnum + 1
        new_bid = f"B-{new_num}"

        # count helpful/harmful from tags
        helpful_count = sum(1 for t in curator_bullet_tags.values() if str(t).lower() == "helpful")
        harmful_count = sum(1 for t in curator_bullet_tags.values() if str(t).lower() == "harmful")

        new_row = {
            "metadata": {
                "bullet_id": new_bid,
                "title": curator_title,
                "content": curator_content,
                "helpful_count": int(helpful_count),
                "harmful_count": int(harmful_count),
                "type": "playbook"
            }
        }

        # pick a numeric next key for metadb (avoid collisions)
        try:
            existing_keys = [int(k) for k in vdb.metadb.keys() if (isinstance(k, int) or (isinstance(k, str) and k.isdigit()))]
            next_key = max(existing_keys) + 1 if existing_keys else 0
        except Exception:
            # fallback: use len
            next_key = max([int(k) for k in vdb.metadb.keys() if isinstance(k, int)] + [-1]) + 1 if vdb.metadb else 0

        vdb.metadb[next_key] = new_row
        try:
            vdb._save()
            playbook_updated = True
        except Exception as e:
            print("Warning: failed to save new playbook entry:", e)
            playbook_updated = False

        iteration.consolidation_output["playbook_update_boolean"] = bool(playbook_updated)
        return state

    # Any other action -> treat as ignore
    iteration.consolidation_output["playbook_update_boolean"] = False
    return state



def human_feedback_node(state: LangGraphState) -> LangGraphState:
    iteration = state.iterations[-1]

    if iteration.curator_output["score"] >= 9:
        iteration.human_feedback_output.update({
            "human_approval_status": True,
            "human_feedback": "Approved by human reviewer."
        })
    else:
        iteration.human_feedback_output.update({
            "human_approval_status": False,
            "human_feedback": "Requires further revision."
        })

    return state


# -------------------- LANGGRAPH FLOW --------------------
def build_ace_graph() -> StateGraph:
    graph = StateGraph(LangGraphState)

    graph.add_node("generator", lambda s: generator_node_with_playbook(s, vdb))
    graph.add_node("reflector", reflector_node)
    graph.add_node("curator", curator_node)
    graph.add_node("consolidator", consolidator_node)
    graph.add_node("human_feedback", human_feedback_node)

    graph.add_edge("generator", "reflector")
    graph.add_edge("reflector", "curator")
    graph.add_edge("curator", "consolidator")

    graph.add_conditional_edges(
    "consolidator",
    lambda state: (
        "human_feedback"
        if state.iterations[-1].curator_output["score"] >= 9
        or len(state.iterations) >= 3
        else "generator"
    ),
    {"human_feedback": END, "generator": "generator"},
)

    graph.set_entry_point("generator")
    return graph


# -------------------- MAIN --------------------
def main():
    ace_graph = build_ace_graph().compile()

    initial_state = LangGraphState(iterations=[
        IterationSnapshot(generator_inputs={"user_task": "Calculate CAGR using Excel formula"})
    ])

    # 🟢 Invoke the compiled graph
    final_state = ace_graph.invoke(initial_state)

    # ✅ Handle both dict and model cases safely
    iterations = final_state.get("iterations") if isinstance(final_state, dict) else final_state.iterations
    last_iter = iterations[-1]

    print("\n=== Last Iteration ===")
    print("Score:", last_iter["curator_output"]["score"] if isinstance(last_iter, dict) else last_iter.curator_output["score"])
    print("Decision:", last_iter["human_feedback_output"]["human_approval_status"] if isinstance(last_iter, dict) else last_iter.human_feedback_output["human_approval_status"])
    print("Curator Feedback:", last_iter["curator_output"]["content"] if isinstance(last_iter, dict) else last_iter.curator_output["content"])

if __name__ == "__main__":
    main()
