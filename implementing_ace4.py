import os
import pickle
import uuid
import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import faiss
import numpy as np
from datetime import datetime

from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver   # ✅ Checkpointing
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()


# ------------------------------------------------------
#                  My Vector Database
# ------------------------------------------------------




# -------------------- CONFIG --------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_FILE_TEMPLATE = "vectors_{name}.npy"       # file per collection
METADATA_FILE_TEMPLATE = "metadata_{name}.pkl"   # metadata per collection

DEFAULT_COLLECTIONS = ["playbook", "states", "system_prompts"]

BATCH_SIZE = 64


# -------------------- Multi-Collection Vector DB --------------------
class MultiCollectionVectorDB:
    """
    Minimal FAISS-backed vector DB that supports multiple named collections.

    Each collection keeps:
      - a faiss.IndexFlatL2 index (in-memory)
      - a numpy file with all vectors (vectors_{name}.npy)
      - a metadata pickle mapping integer ids -> payload dict (metadata_{name}.pkl)

    Payload (metadata) is expected to contain at least:
      - 'Bullet_ID' or other id-like fields when dealing with playbook
      - 'text' field used for embedding
    """

    def __init__(self, collections: Optional[List[str]] = None, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.collections: Dict[str, Dict[str, Any]] = {}  # name -> {"index": faiss_index, "metadata": dict}
        self._init_collections(collections or DEFAULT_COLLECTIONS)

    def _init_collections(self, names: List[str]):
        for name in names:
            self.collections[name] = {"index": None, "metadata": {}}
            self._load_collection(name)

    def _vector_path(self, name: str) -> str:
        return VECTOR_FILE_TEMPLATE.format(name=name)

    def _meta_path(self, name: str) -> str:
        return METADATA_FILE_TEMPLATE.format(name=name)

    def _load_collection(self, name: str):
        vect_path = self._vector_path(name)
        meta_path = self._meta_path(name)

        if os.path.exists(vect_path) and os.path.exists(meta_path):
            xb = np.load(vect_path)
            d = xb.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(xb)
            with open(meta_path, "rb") as f:
                metadb = pickle.load(f)
            self.collections[name]["index"] = index
            self.collections[name]["metadata"] = metadb
            print(f"Loaded collection '{name}' with {index.ntotal} vectors (dim={d}).")
        else:
            # initialize empty
            self.collections[name]["index"] = None
            self.collections[name]["metadata"] = {}
            print(f"Initialized empty collection '{name}'.")

    def _save_collection(self, name: str):
        vect_path = self._vector_path(name)
        meta_path = self._meta_path(name)

        # persist metadata
        with open(meta_path, "wb") as f:
            pickle.dump(self.collections[name]["metadata"], f)

        # persist vectors by re-embedding all texts in the metadata (keeps canonical order)
        if len(self.collections[name]["metadata"]) > 0:
            ordered_keys = sorted(self.collections[name]["metadata"].keys())
            texts = [self.collections[name]["metadata"][k]["text"] for k in ordered_keys]
            xb = self.embed(texts)
            np.save(vect_path, xb)
            # also rewrite faiss index to reflect xb
            d = xb.shape[1]
            ix = faiss.IndexFlatL2(d)
            ix.add(xb)
            self.collections[name]["index"] = ix
        else:
            # remove files if exist
            if os.path.exists(vect_path):
                os.remove(vect_path)
        print(f"Persisted collection '{name}'. files: {vect_path}, {meta_path}")

    def persist(self, name: Optional[str] = None):
        """Persist a single collection or all if name is None."""
        if name:
            if name not in self.collections:
                raise KeyError(f"Collection '{name}' not found")
            self._save_collection(name)
        else:
            for n in list(self.collections.keys()):
                self._save_collection(n)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return float32 numpy array of embeddings for input texts."""
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # convert to float32 (faiss expects float32)
        return np.asarray(embs, dtype=np.float32)

    def _ensure_index_for(self, name: str, dim: int):
        if self.collections[name]["index"] is None:
            self.collections[name]["index"] = faiss.IndexFlatL2(dim)

    def list_collections(self) -> List[str]:
        return list(self.collections.keys())

    def get_collection_size(self, name: str) -> int:
        idx = self.collections[name]["index"]
        return 0 if idx is None else int(idx.ntotal)

    def upsert(self, name: str, docs: List[Dict[str, Any]]):
        """
        Upsert docs into named collection.
        Each doc must be a dict that contains at least:
          - 'text' (string) : the content to embed
          - any additional payload fields (stored in metadata)

        This implementation appends new vectors and assigns integer internal ids
        (0-based per-collection). Metadata keys are those integer ids.
        """
        if name not in self.collections:
            # lazily create collection
            self.collections[name] = {"index": None, "metadata": {}}

        texts = [d["text"] for d in docs]
        xs = self.embed(texts)
        dim = xs.shape[1]
        self._ensure_index_for(name, dim)
        index = self.collections[name]["index"]

        start_idx = index.ntotal if index is not None else 0
        index.add(xs)

        # add metadata mapping from new integer ids to payload
        for i, d in enumerate(docs):
            key = start_idx + i
            # copy payload but ensure 'text' present
            payload = dict(d)
            payload.setdefault("text", texts[i])
            self.collections[name]["metadata"][key] = payload

        # persist only vectors appended incrementally by saving full set
        self._save_collection(name)
        print(f"Upserted {len(docs)} docs into '{name}'. New size: {self.get_collection_size(name)}")


    def get_prompt_by_node(self, node_name: str):
        meta = self.collections["system_prompts"]["metadata"]
        for _, v in meta.items():
            if v.get("node_name") == node_name:
                return v
        return None

    def replace_prompt(self, node_name: str, new_text: str):
   
        if "system_prompts" not in self.collections:
            raise KeyError("Collection 'system_prompts' not found")

        meta = self.collections["system_prompts"]["metadata"]

        # remove all existing entries for this node
        to_delete = [k for k, v in meta.items() if v.get("node_name") == node_name]
        for k in to_delete:
            del meta[k]

        # insert new record with a fresh integer key
        new_id = max(meta.keys(), default=-1) + 1
        meta[new_id] = {
            "node_name": node_name,
            "type": "static",
            "text": new_text,
            "last_updated": datetime.now().isoformat()
        }

    # IMPORTANT: rebuild vectors from *current* metadata only
        self._save_collection("system_prompts")

        
    def query(self, name: str, query_text: str, top_k: int = 3):
        """Return list of (metadata, distance) tuples for the top_k nearest matches."""
        if name not in self.collections:
            raise KeyError(f"Collection '{name}' not found")
        index = self.collections[name]["index"]
        if index is None or index.ntotal == 0:
            return []

        qvec = self.embed([query_text])
        D, I = index.search(qvec, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            md = self.collections[name]["metadata"].get(idx, {})
            results.append({"metadata": md, "distance": float(dist), "internal_id": int(idx)})
        return results

# vdb = MultiCollectionVectorDB()




# --------------------------------------------------------------------
# -------------------- SAFE STATE DEFINITIONS ------------------------
# --------------------------------------------------------------------

class LangGraphState(BaseModel):
    # === Core identifiers ===
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # === Metadata ===
    metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "task_name": "",
        "iteration": 0,
        "score": 0.0
    })

    # === Control fields ===
    # visited: List[str] = Field(default_factory=list)
    run_human_feedback: bool = False
    is_iteration_complete: bool = False
    # current_route: str = ""
    

    # === Generator ===
    generator_inputs: Dict[str, Any] = Field(default_factory=lambda: {
        "user_task": ""
    })
    generator_output: Dict[str, Any] = Field(default_factory=lambda: {
        "final_ans": "",
        "reasoning": "",
        "bullet_context": []
    })

    # === Reflector ===
    reflector_output: Dict[str, Any] = Field(default_factory=lambda: {
        "reasoning": "",
        "error_identification": [],
        "rootcauseanalysis": "",
        "correctapproach": "",
        "keyinsights": [],
        "bullet_tags": {}
    })

    # === Curator ===
    curator_output: Dict[str, Any] = Field(default_factory=lambda: {
        "action": [],
        "content": "",
        "title": "",
        "score": 0.0,
        "bullet_tags": {}
    })

    # === Consolidator ===
    consolidation_output: Dict[str, Any] = Field(default_factory=lambda: {
        "playbook_update_boolean": False,
        "generator_retry_count": 0,
        "route": ""
    })

    # === Human Feedback ===
    human_feedback_output: Dict[str, Any] = Field(default_factory=lambda: {
        "human_approval_status": None,
        "human_feedback": None
    })

# --------------------------------------------------------------------
# -------------------- LLM SETUP ------------------------------------
# --------------------------------------------------------------------

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


#  --------------------------------------------------------
#                     Helper Functions
# ---------------------------------------------------------

def fetch_all_from_vdb(vdb, collection: str) -> List[Dict[str, Any]]:
    """
    Return all stored records from the multi-collection vdb as a list of dicts,
    sorted by internal integer id. Each dict is the payload stored in metadata.
    """
    records = []
    if not hasattr(vdb, "collections") or collection not in vdb.collections:
        return records

    metadb = vdb.collections[collection].get("metadata", {})
    for internal_id in sorted(metadb.keys()):
        # ensure we return a copy to avoid accidental mutation
        payload = dict(metadb[internal_id])
        payload["_internal_id"] = int(internal_id)
        records.append(payload)
    return records



def fetch_full_playbook_from_vdb(vdb, collection: str) -> List[Dict[str, Any]]:
    """
    Return a list of playbook records (canonical dicts) from the specified collection.
    Detects playbook entries by presence of 'Bullet_ID'/'bullet_id' or where a 'type' == 'playbook'.
    """
    recs = []
    if not hasattr(vdb, "collections") or collection not in vdb.collections:
        return recs

    metadb = vdb.collections[collection].get("metadata", {})
    for internal_id in sorted(metadb.keys()):
        row = metadb[internal_id]

        # normalize keys
        row_type = (row.get("type") or row.get("Type") or "").lower()
        bullet_id = row.get("Bullet_ID") or row.get("bullet_id")

        is_playbook = (
            (isinstance(row_type, str) and row_type == "playbook")
        )

        if is_playbook:
            canonical = {
                "internal_id": int(internal_id),
                "bullet_id": bullet_id,
                "title": row.get("Title") or row.get("title") or "",
                "content": row.get("Content") or row.get("content") or row.get("text") or "",
                # ✅ unified field names
                "helpful_count": int(row.get("helpful_count", row.get("helpful_content", 0) or 0)),
                "harmful_count": int(row.get("harmful_count", row.get("harmful_content", 0) or 0)),
                "raw": row
            }
            recs.append(canonical)

    return recs



def format_playbook_for_prompt(playbook_rows: List[Dict[str, Any]]) -> str:
    """
    Formats playbook entries clearly for LLM use.
    Each bullet is labeled with consistent tags (BULLET_ID, TITLE, CONTENT),
    making it easier for the LLM to reference them in BULLET_CONTEXT.
    """
    if not playbook_rows:
        return "No playbook bullets found."

    formatted = []
    for p in playbook_rows:
        bullet_id = p.get("bullet_id") or "UNKNOWN_ID"
        title = (p.get("title") or "").strip()
        content = (p.get("content") or "").strip()
        helpful = p.get("helpful_count", 0)
        harmful = p.get("harmful_count", 0)

        formatted.append(
            f"BULLET_ID: {bullet_id}\n"
            f"TITLE: {title}\n"
            f"CONTENT: {content}\n"
            f"(helpful={helpful}, harmful={harmful})"
        )

    return "\n\n".join(formatted)


def get_or_update_system_prompt(vdb: MultiCollectionVectorDB, node_name: str) -> str:
    record = vdb.get_prompt_by_node(node_name)
    static_prompt = record.get("text", "") if record else ""

    if record:
        print(f"\n--- Static system prompt for node '{node_name}' ---\n")
        print(static_prompt)
        print("\n-----------------------------------------------\n")
    else:
        print(f"No system prompt found for node '{node_name}'. Initializing blank.\n")

    choice = input("Do you want to update the system prompt for this node? (y/n): ").strip().lower()
    if choice == "y":
        print("Enter the new system prompt (end with 'EOF' on a new line):")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "EOF":
                break
            lines.append(line)
        new_prompt = "\n".join(lines)

        vdb.replace_prompt(node_name, new_prompt)
        print(f"✅ Updated system prompt for node '{node_name}'.\n")
        return new_prompt

    return static_prompt



# def parse_bullet_context(text: str):
#         txt = (text or "").strip()
#         if not txt or txt.upper() == "NONE":
#             return []

#         # 1) try strict JSON
#         try:
#             parsed = json.loads(txt)
#             out = []
#             if isinstance(parsed, list):
#                 for item in parsed:
#                     if isinstance(item, list) and len(item) >= 1:
#                         bid = str(item[0]).strip()
#                         title = str(item[1]).strip() if len(item) > 1 else ""
#                         content = str(item[2]).strip() if len(item) > 2 else ""
#                         out.append({"id": bid, "title": title, "content": content})
#                 if out:
#                     return out
#         except Exception:
#             pass

#         # 2) attempt light coercion (single->double quotes, quote B-IDs)
#         try:
#             cand = txt.replace("'", '"')
#             cand = re.sub(r'\[([^\[\]]*?)\]', lambda m: "[" + re.sub(r'(\bB-\d+\b)', r'"\1"', m.group(1)) + "]", cand)
#             parsed = json.loads(cand)
#             out = []
#             if isinstance(parsed, list):
#                 for item in parsed:
#                     if isinstance(item, list) and len(item) >= 1:
#                         bid = str(item[0]).strip()
#                         title = str(item[1]).strip() if len(item) > 1 else ""
#                         content = str(item[2]).strip() if len(item) > 2 else ""
#                         out.append({"id": bid, "title": title, "content": content})
#                 if out:
#                     return out
#         except Exception:
#             pass

#         # 3) regex: extract bracketed rows like [B-101, title101, content101]
#         rows = re.findall(r'\[([^\[\]]+?)\]', txt)
#         parsed_rows = []
#         for r in rows:
#             parts = [p.strip() for p in re.split(r'\s*,\s*', r)]
#             if len(parts) >= 1:
#                 bid = parts[0]
#                 title = parts[1] if len(parts) > 1 else ""
#                 content = parts[2] if len(parts) > 2 else ""
#                 parsed_rows.append({"id": bid, "title": title, "content": content})
#         if parsed_rows:
#             return parsed_rows

#         # 4) fallback: extract bare B-IDs
#         ids = re.findall(r'(B-\d+)', txt, flags=re.IGNORECASE)
#         return [{"id": i.upper(), "title": "", "content": ""} for i in ids]


def parse_bullet_context(text: str):
    txt = (text or "").strip()
    if not txt or txt.upper() == "NONE":
        return []

    # Extract all sections like [bullet id - PB0004, title - "...", content - "..."]
    pattern = re.compile(
        r'bullet id\s*-\s*([A-Za-z0-9_-]+)\s*,\s*'
        r'title\s*-\s*"?([^,"]+?)"?\s*,\s*'
        r'content\s*-\s*"?(.+?)"?(?:\]|\n|$)',
        re.IGNORECASE | re.DOTALL
    )

    matches = pattern.findall(txt)
    results = []
    for match in matches:
        bid, title, content = [m.strip() for m in match]
        results.append({
            "id": bid,
            "title": title,
            "content": content
        })

    return results


def generator_node_with_playbook(
    state: LangGraphState,
    vdb,  # instance of MultiCollectionVectorDB
    playbook_override: Optional[List[Dict[str, Any]]] = None,
    memory_collection: str = "states",
    playbook_collection: str = "playbook"
) -> LangGraphState:
    """
    Generator node adapted for flattened checkpoint-based LangGraphState.
    - Fetches memory & playbook from vector DB
    - Generates new output and updates control fields
    - Prepares state for the next node (Reflector)
    """
    # Fetch static part (with optional update)
    static_prompt = get_or_update_system_prompt(vdb, "Generator")
    # ------------------------
    # 🔹 Derive iteration + task info
    # ------------------------
    iteration_num = state.metadata.get("iteration", 0) + 1
    task_id = state.task_id
    user_task = state.generator_inputs.get("user_task", "")

    # print(f"\n🧠 Running GENERATOR for task_id={task_id} | iteration={iteration_num}")
    # print(f"Task: {user_task}")

    # ------------------------
    # 🔹 Fetch memory (previous completed states) from VDB
    # ------------------------
    try:
        all_records = fetch_all_from_vdb(vdb, memory_collection)
    except Exception:
        all_records = []

    summaries = []
    for rec in all_records:
        # each rec corresponds to a completed iteration checkpoint
        # summaries.append({
        #     "task_id": rec.get("task_id", ""),
        #     "iteration": rec.get("metadata", {}).get("iteration", ""),
        #     "task_name": rec.get("metadata", {}).get("task_name", ""),
        #     "score": rec.get("metadata", {}).get("score", ""),
        #     "user_task": rec.get("generator_inputs", {}).get("user_task", ""),
        #     "final_ans": rec.get("generator_output", {}).get("final_ans", ""),
        #     "human_approval_status": rec.get("human_feedback_output", {}).get("human_approval_status", None),
        #     "human_feedback": rec.get("human_feedback_output", {}).get("human_feedback", None)
        # })

        for itr in rec.get("iterations", []):
            summaries.append({
                "task_id": rec.get("task_id", ""),
                "iteration": itr["metadata"].get("iteration", ""),
                "task_name": itr["metadata"].get("task_name", ""),
                "score": itr["metadata"].get("score", ""),
                "user_task": itr["generator_inputs"].get("user_task", ""),
                "final_ans": itr["generator_output"].get("final_ans", ""),
                "human_approval_status": itr["human_feedback_output"].get("human_approval_status", None),
                "human_feedback": itr["human_feedback_output"].get("human_feedback", None)
            })

    if not summaries:
        memory_summary = "No prior approved tasks found in memory."
    else:
        lines = ["### Prior Approved Tasks and Answers ###\n"]
        for s in summaries:
            approval = "✅ Approved" if s.get("human_approval_status") else "❌ Not Approved"
            lines.append(
                f"- **Task ID:** {s['task_id']}\n"
                f"  • **Task Name:** {s['task_name']}\n"
                f"  • **User Request:** {s['user_task']}\n"
                f"  • **Final Answer:** {s['final_ans']}\n"
                f"  • **Score:** {s['score']}\n"
                f"  • **Human Feedback:** {s['human_feedback']} | {approval}\n"
            )
        memory_summary = "\n".join(lines)

    # ------------------------
    # 🔹 Fetch playbook bullets
    # ------------------------
    if playbook_override is not None:
        playbook_rows = playbook_override
    else:
        try:
            playbook_rows = fetch_full_playbook_from_vdb(vdb, playbook_collection)
        except Exception:
            playbook_rows = []

    playbook_text = format_playbook_for_prompt(playbook_rows)

    # ------------------------
    # 🔹 Build prompt for LLM
    # ------------------------
    
    dynamic_prompt = f"""
        PLAYBOOK (choose from these bullets):
        {playbook_text}

        FULL MEMORY:
        {memory_summary}

        Now produce all labeled sections for task: {user_task}
        """

    system_prompt = f"{static_prompt}\n{dynamic_prompt}"

    print("YEH GENERATOR KA SYSTEM PROMPT HH", system_prompt)
    print("-" * 90)

    user_prompt = f"Task: {user_task}"
    raw_output = call_llm(system_prompt, user_prompt)
    normalized = (raw_output or "").replace("\r\n", "\n")

    print("YEH GENERATOR KA LLM OUTPUT HH", normalized)
    print("-" * 90)

    # ------------------------
    # 🔹 Extract labeled sections from LLM output
    # ------------------------
    def extract_section_by_labels(text: str, start_label: str, next_labels: List[str]) -> str:
        s_idx = text.find(start_label)
        if s_idx < 0:
            return ""
        s_idx += len(start_label)
        next_positions = [text.find(lbl, s_idx) for lbl in next_labels if text.find(lbl, s_idx) >= 0]
        if next_positions:
            end_idx = min(next_positions)
            return text[s_idx:end_idx].strip()
        else:
            return text[s_idx:].strip()

    labels = ["FINAL_ANS:", "REASONING:", "BULLETS_USED:", "BULLET_CONTEXT:"]
    final_ans_text = extract_section_by_labels(normalized, "FINAL_ANS:", labels[1:])
    reasoning_text = extract_section_by_labels(normalized, "REASONING:", labels[2:])
    bullets_text = extract_section_by_labels(normalized, "BULLETS_USED:", ["BULLET_CONTEXT:"])
    bullet_context_text = extract_section_by_labels(normalized, "BULLET_CONTEXT:", [])

    parsed_bullet_context = parse_bullet_context(bullet_context_text)

    # ------------------------
    # 🔹 Update flattened state
    # ------------------------
    state.metadata["iteration"] = iteration_num
    state.metadata["task_name"] = user_task
    # state.visited.append("Generator")
    # state.current_route = "Generator"
    state.is_iteration_complete = False  # iteration continues

    # overwrite generator output fields
    state.generator_output = {
        "final_ans": final_ans_text,
        "reasoning": reasoning_text,
        "bullet_context": parsed_bullet_context
    }
    # state.generator_output = json.loads(json.dumps(state.generator_output))

    # print(f"✅ Generator node completed iteration {iteration_num}")
    return state



def reflector_node(state: "LangGraphState", vdb) -> "LangGraphState":
    """
    Reflector node adapted for the flattened, checkpoint-based LangGraphState.
    - Reads prior examples (completed iterations) from the 'states' collection.
    - Analyzes the current generator output stored directly in state.
    - Calls LLM with JSON schema instructions and parses response robustly.
    - Updates `reflector_output` and control fields for checkpointing.
    """

    # -----------------------------
    # 1️⃣ Context from current state
    # -----------------------------
    gen_final = state.generator_output.get("final_ans", "")
    gen_reasoning = state.generator_output.get("reasoning", "")
    gen_bullets = state.generator_output.get("bullet_context", []) or []
    user_task = state.generator_inputs.get("user_task", "")
    iteration_num = state.metadata.get("iteration", 0)
    # task_id = state.task_id

    # print(f"\n🧩 Running REFLECTOR for task_id={task_id} | iteration={iteration_num}")
    # print(f"User Task: {user_task}")

    # ----------------------------ca-
    # 2️⃣ Fetch prior reflector insights from VDB
    # -----------------------------
    try:
        all_rows = fetch_all_from_vdb(vdb, "states")
    except Exception:
        all_rows = []

    summaries = []
    for rec in all_rows:
        # meta = rec.get("metadata", {})
        # gen_out = rec.get("generator_output", {})
        # refl_out = rec.get("reflector_output", {})
        # human_fb = rec.get("human_feedback_output", {})
        for itr in rec.get("iterations", []):
            meta = itr.get("metadata", {})
            gen_out = itr.get("generator_output", {})
            refl_out = itr.get("reflector_output", {})
            human_fb = itr.get("human_feedback_output", {})

        summaries.append({
            "task_id": rec.get("task_id", ""),
            "iteration": meta.get("iteration", ""),
            "task_name": meta.get("task_name", ""),
            "approval_status": human_fb.get("human_approval_status", None),
            "generator_final_ans": gen_out.get("final_ans", ""),
            "reflector_reasoning": refl_out.get("reasoning", ""),
            "reflector_error_identification": refl_out.get("error_identification", []),
            "reflector_rootcauseanalysis": refl_out.get("rootcauseanalysis", ""),
            "reflector_keyinsights": refl_out.get("keyinsights", []),
            "reflector_correctapproach": refl_out.get("correctapproach", "")
        })


    #     summaries = []
    # for rec in all_rows:
    #     task_id = rec.get("task_id", "")
    #     for itr in rec.get("iterations", []):
    #         meta = itr.get("metadata", {})
    #         gen_out = itr.get("generator_output", {})
    #         refl_out = itr.get("reflector_output", {})
    #         human_fb = itr.get("human_feedback_output", {})

    #         summaries.append({
    #             "task_id": task_id,
    #             "iteration": meta.get("iteration", ""),
    #             "task_name": meta.get("task_name", ""),
    #             "approval_status": human_fb.get("human_approval_status", None),
    #             "generator_final_ans": gen_out.get("final_ans", ""),
    #             "reflector_reasoning": refl_out.get("reasoning", ""),
    #             "reflector_error_identification": refl_out.get("error_identification", []),
    #             "reflector_rootcauseanalysis": refl_out.get("rootcauseanalysis", ""),
    #             "reflector_keyinsights": refl_out.get("keyinsights", []),
    #             "reflector_correctapproach": refl_out.get("correctapproach", "")
    #         })

    # -----------------------------
    # 3️⃣ Build compact memory summary
    # -----------------------------
    if not summaries:
        print("⚠️ No prior reflector insights found — running with empty memory.")
        memory_summary = (
            "⚠️ No prior reflector insights found in memory.\n"
            "Use only the current generator output for analysis.\n"
        )
    else:
        lines = ["### Prior Task Reflector Insights ###\n"]
        for s in summaries:
            approval = "✅ Approved" if s.get("approval_status") else "❌ Not Approved"
            lines.append(
                f"- **Task ID:** {s['task_id']} | Iteration: {s['iteration']}\n"
                f"  • **Task Name:** {s['task_name']}\n"
                f"  • **Final Answer:** {s['generator_final_ans']}\n"
                f"  • **Reasoning:** {s['reflector_reasoning']}\n"
                f"  • **Errors Identified:** {s['reflector_error_identification']}\n"
                f"  • **Root Cause:** {s['reflector_rootcauseanalysis']}\n"
                f"  • **Correct Approach:** {s['reflector_correctapproach']}\n"
                f"  • **Key Insights:** {s['reflector_keyinsights']}\n"
                f"  • **Approval:** {approval}\n"
            )
        memory_summary = "\n".join(lines)

    # -----------------------------
    # 4️⃣ Prepare LLM prompt
    # -----------------------------
    # schema_instructions = (
    #         "Return JSON with the following top-level keys:\n"
    #         "{\n"
    #         '  "reflector_output": {\n'
    #         '      "reasoning": string,\n'
    #         '      "error_identification": [string],\n'
    #         '      "rootcauseanalysis": string,\n'
    #         '      "correctapproach": string,\n'
    #         '      "keyinsights": [string],\n'
    #         '      "bullet_tags": { "<bullet_id>": "helpful" | "harmful", ... }\n'
    #         '  }\n'
    #         '}\n'
    #     )

    schema_instructions = (
        "Return JSON with the following top-level keys:\n"
        "{\n"
        '  "reflector_output": {\n'
        '      "reasoning": string,\n'
        '      "error_identification": [string],\n'
        '      "rootcauseanalysis": string,\n'
        '      "correctapproach": string,\n'
        '      "keyinsights": [string],\n'
        '      "bullet_tags": { "<bullet_id>": "helpful" | "harmful", ... }\n'
        '  }\n'
        '}\n'
    )
    static_prompt = get_or_update_system_prompt(vdb, "Reflector")
    dynamic_prompt = f"""
        SCHEMA EXPECTED:
        {schema_instructions}

        HISTORY (compact previews):
        {memory_summary}

        IMPORTANT: If analysis is partial or uncertain, still return valid JSON with placeholders.
        """

    system_prompt = f"{static_prompt}\n\n{dynamic_prompt}"

    user_prompt = (
        "Present generator output (primary context):\n"
        f"FINAL_ANS:\n{gen_final}\n\n"
        f"REASONING:\n{gen_reasoning}\n\n"
        f"BULLET_CONTEXT:\n{json.dumps(gen_bullets, ensure_ascii=False)}\n\n"
        f"TASK INPUT:\n{user_task}\n\n"
        "Analyze and return JSON as instructed in the system message."
    )


    print("🧠 REFLECTOR SYSTEM PROMPT:\n", system_prompt)
    print("-" * 60)
    print("🧠 REFLECTOR USER PROMPT:\n", user_prompt)
    print("-" * 60)

    # -----------------------------
    # 5️⃣ Call LLM and parse JSON
    # -----------------------------
    raw_response = call_llm(system_prompt, user_prompt)
    print("LLM RAW RESPONSE (Reflector):", raw_response)
    print("-" * 80)

    parsed = None
    try:
        parsed = json.loads(raw_response)
    except Exception:
        try:
            start = raw_response.index("{")
            end = raw_response.rindex("}") + 1
            parsed = json.loads(raw_response[start:end])
        except Exception:
            parsed = None

    # -----------------------------
    # 6️⃣ Update reflector_output safely
    # -----------------------------
    if isinstance(parsed, dict) and "reflector_output" in parsed:
        state.reflector_output = parsed["reflector_output"]
    else:
        # Fallback if parsing failed
        normalized_ids = []
        for item in gen_bullets or []:
            if isinstance(item, dict) and item.get("id"):
                normalized_ids.append(item["id"].strip().upper())
        seen = set()
        dedup_ids = [i for i in normalized_ids if not (i in seen or seen.add(i))]

        state.reflector_output.update({
            "reasoning": raw_response,
            "error_identification": ["analysis_not_parsed_from_llm"],
            "rootcauseanalysis": "",
            "correctapproach": "",
            "keyinsights": [],
            "bullet_tags": {bid: "harmful" for bid in dedup_ids}
        })

    # -----------------------------
    # 7️⃣ Update control fields for checkpointing
    # -----------------------------
    # state.visited.append("Reflector")
    # state.current_route = "Reflector"
    state.is_iteration_complete = False  # still in progress

    # print(f"✅ Reflector node completed iteration {iteration_num}")
    return state



def curator_node(state: LangGraphState, vdb) -> LangGraphState:
    """
    Curator node adapted for the flattened, checkpoint-based LangGraphState.
    - Reads playbook and memory from the MultiCollectionVectorDB.
    - Analyzes generator + reflector outputs.
    - Calls LLM to decide whether to add/update/ignore a playbook bullet.
    - Updates state.curator_output and metadata.score.
    """

    # -----------------------------
    # 1️⃣ Extract current context from state
    # -----------------------------
    task_id = state.task_id
    iteration_num = state.metadata.get("iteration", 0)
    task_name = state.metadata.get("task_name", "")
    gen_final_ans = state.generator_output.get("final_ans", "")
    refl_reasoning = state.reflector_output.get("reasoning", "")
    refl_errors = state.reflector_output.get("error_identification", [])
    refl_rootcause = state.reflector_output.get("rootcauseanalysis", "")
    refl_correct = state.reflector_output.get("correctapproach", "")
    refl_keyinsights = state.reflector_output.get("keyinsights", [])
    # refl_bullet_tags = state.reflector_output.get("bullet_tags", {})

    # print(f"\n🧠 Running CURATOR for task_id={task_id} | iteration={iteration_num}")
    # print(f"Task Name: {task_name}")

    # -----------------------------
    # 2️⃣ Fetch prior curator memory from 'states' collection
    # -----------------------------
    try:
        all_rows = fetch_all_from_vdb(vdb, "states")
    except Exception:
        all_rows = []

    # summaries = []
    # for rec in all_rows:
    #     meta = rec.get("metadata", {})
    #     cur_out = rec.get("curator_output", {})
    #     human_fb = rec.get("human_feedback_output", {})
    #     summaries.append({
    #         "task_id": rec.get("task_id", ""),
    #         "iteration": meta.get("iteration", ""),
    #         "score": meta.get("score", ""),
    #         "approval_status": human_fb.get("human_approval_status", None),
    #         "curator_action": cur_out.get("action", []),
    #         "curator_content": cur_out.get("content", "")
    #     })

    summaries = []
    for rec in all_rows:
        task_id = rec.get("task_id", "")
        for itr in rec.get("iterations", []):
            meta = itr.get("metadata", {})
            cur_out = itr.get("curator_output", {})
            human_fb = itr.get("human_feedback_output", {})

            summaries.append({
                "task_id": task_id,
                "iteration": meta.get("iteration", ""),
                "score": meta.get("score", ""),
                "approval_status": human_fb.get("human_approval_status", None),
                "curator_action": cur_out.get("action", []),
                "curator_content": cur_out.get("content", "")
            })

    if not summaries:
        print("⚠️ No prior curator insights found — running with empty memory.")
        memory_summary = (
            "⚠️ No prior curator insights found in memory.\n"
            "Use only current generator and reflector outputs.\n"
        )
    else:
        lines = ["### Prior Task Curator Insights ###\n"]
        for s in summaries:
            approval = "✅ Approved" if s.get("approval_status") else "❌ Not Approved"
            lines.append(
                f"- **Task ID:** {s['task_id']} | Iteration: {s['iteration']}\n"
                f"  • **Curator Action:** {s['curator_action']}\n"
                f"  • **Curator Content:** {s['curator_content']}\n"
                f"  • **Score:** {s['score']}\n"
                f"  • **Approval:** {approval}\n"
            )
        memory_summary = "\n".join(lines)

    # -----------------------------
    # 3️⃣ Fetch and format playbook bullets
    # -----------------------------
    try:
        playbook_rows = fetch_full_playbook_from_vdb(vdb, "playbook") if hasattr(vdb, "collections") else []
    except Exception:
        playbook_rows = []
    playbook_text = format_playbook_for_prompt(playbook_rows) if playbook_rows else "No playbook bullets found."

    # -----------------------------
    # 4️⃣ Build LLM schema + prompts
    # -----------------------------
    schema_instructions = (
        "Return ONLY a valid JSON object (no extra commentary) using this exact structure:\n\n"
        "{\n"
        '  "metadata_task_name": string|null,\n'
        '  "generator_final_ans": string,\n'
        '  "curator_output": {\n'
        '      "action": "add" | "update" | "ignore",\n'
        '      "title": string|null,\n'
        '      "content": string|null,\n'
        '      "bullet_id": string|null,\n'
        '      "score": number\n'
        '  }\n'
        '}\n'
    )

    # -----------------------------
    # 🧩 SYSTEM PROMPT
    # -----------------------------
    static_prompt = get_or_update_system_prompt(vdb, "Curator")

    dynamic_prompt = f"""
        {schema_instructions}

        PLAYBOOK (entire):
        {playbook_text}

        MEMORY (states collection):
        {memory_summary}
        """
    system_prompt = f"{static_prompt}\n\n{dynamic_prompt}"

    print("YEH CURATOR KA SYSTEM PROMPT HH:", system_prompt)


    user_prompt = (
        "Present context (from Reflector + Generator):\n"
        f"Task name: {task_name}\n\n"
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
        "Using the full playbook and the memory above, decide whether the candidate insight should be added, used to refine an existing bullet (update), or ignored.\n"
        "Return the JSON described in the system message."
    )
    
    print("YEHH CURATOR KA USER PROMPT:\n", user_prompt)


    # -----------------------------
    # 5️⃣ Call LLM and parse JSON
    # -----------------------------
    raw = call_llm(system_prompt, user_prompt)
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

    print("CURATOR RAW RESPONSE:", parsed)
    print("-" * 80)

    # -----------------------------
    # 6️⃣ Extract curator output safely
    # -----------------------------
    if isinstance(parsed, dict) and "curator_output" in parsed:
        curator_block = parsed["curator_output"]
    else:
        curator_block = {"action": "ignore", "title": None, "content": None, "bullet_id": None, "score": 0}

    final_action = curator_block.get("action", "ignore")
    title = curator_block.get("title")
    content = curator_block.get("content")
    bullet_id = curator_block.get("bullet_id")
    score = curator_block.get("score", 0.0)

    # -----------------------------
    # 7️⃣ Update flattened state fields
    # -----------------------------
    state.curator_output = {
        "action": [final_action],
        "content": content,
        "title": title,
        "bullet_id": bullet_id,
        "score": score
    }
    state.metadata["score"] = score

    # -----------------------------
    # 8️⃣ Control flags for checkpointing
    # -----------------------------
    # state.visited.append("Curator")
    # state.current_route = "Curator"
    state.is_iteration_complete = False  # not done yet; next Consolidator/HumanFeedback

    # print(f"✅ Curator node completed iteration {iteration_num}")
    return state


def get_next_bullet_id(prefix="PB", tracker_path="playbook_bullet.json"):
    """Generates sequential playbook bullet IDs (PB0001, PB0002, etc.)."""
    if not os.path.exists(tracker_path):
        with open(tracker_path, "w") as f:
            json.dump({"Bullet_ID": 0}, f)

    with open(tracker_path, "r+") as f:
        data = json.load(f)
        data["Bullet_ID"] += 1
        next_id = f"{prefix}{data['Bullet_ID']:04d}"
        f.seek(0)
        json.dump(data, f)
        f.truncate()
    return next_id


def consolidator_node(state: LangGraphState, vdb) -> LangGraphState:
    """
    🧭 Consolidator Node (checkpoint-based version)

    Responsibilities:
      • Decides routing after Curator:
          - If curator score >= 9 → Human Feedback
          - If curator score < 9 → retry Generator (max 3 times) → then Human Feedback
      • Manages playbook updates:
          - Adds new bullet or updates existing one
          - Increments helpful/harmful counts from Reflector tags
      • Marks iteration as complete when routing to HumanFeedback
    """

    # -----------------------------
    # 1️⃣ Extract fields from current flattened state
    # -----------------------------
    curator_action = state.curator_output.get("action", [])
    if isinstance(curator_action, str):
        curator_action = [curator_action]

    curator_action_lc = [a.strip().lower() for a in curator_action]
    action = curator_action_lc[0] if curator_action_lc else "ignore"

    title = (state.curator_output.get("title") or "").strip()
    content = (state.curator_output.get("content") or "").strip()
    bullet_id = state.curator_output.get("bullet_id") or None
    bullet_tags = state.reflector_output.get("bullet_tags", {}) or {}

    try:
        score = float(state.curator_output.get("score", 0.0))
    except Exception:
        score = 0.0

    task_id = state.task_id
    route = "ignore"
    playbook_updated = False

    print(f"\n🧩 Running CONSOLIDATOR for task_id={task_id} | score={score}")

    # -----------------------------
    # 2️⃣ Decide routing (retry logic)
    # -----------------------------
    generator_retry_count_before = state.consolidation_output.get("generator_retry_count", 0)
    generator_retry_count_after = generator_retry_count_before
    max_retries = 3

    if score >= 9.0:
        route = "human_feedback"
    elif generator_retry_count_before < max_retries:
        route = "generator"
        generator_retry_count_after += 1
    else:
        route = "human_feedback"

    # -----------------------------
    # 3️⃣ Define helper functions
    # -----------------------------
    def ensure_counters(meta: Dict[str, Any]):
        """Ensure helpful/harmful counters exist."""
        if not isinstance(meta, dict):
            return meta
        meta.setdefault("helpful_count", int(meta.get("helpful_content", 0)))
        meta.setdefault("harmful_count", int(meta.get("harmful_content", 0)))
        return meta

    def apply_tags(meta: Dict[str, Any], tags: Dict[str, Any]):
        """Apply helpful/harmful tag increments."""
        ensure_counters(meta)
        for _, v in (tags or {}).items():
            v = str(v).lower()
            if v == "helpful":
                meta["helpful_count"] += 1
            elif v == "harmful":
                meta["harmful_count"] += 1
        return meta

    # -----------------------------
    # 4️⃣ Smart Playbook Update Logic
    # -----------------------------
    try:
        helpful_count = sum(1 for t in bullet_tags.values() if str(t).lower() == "helpful")
        harmful_count = sum(1 for t in bullet_tags.values() if str(t).lower() == "harmful")

        if action in ("ignore", ""):
            playbook_updated = False

        elif action == "update" and bullet_id:
            # --- Update existing playbook entry ---
            playbook_meta = vdb.collections["playbook"].get("metadata", {}) or {}
            bid_upper = bullet_id.strip().upper()
            internal_to_update = None

            for internal_id, payload in playbook_meta.items():
                candidate = str(payload.get("Bullet_ID") or payload.get("bullet_id") or "").strip().upper()
                if candidate == bid_upper:
                    internal_to_update = internal_id
                    break

            if internal_to_update is not None:
                meta = playbook_meta[internal_to_update]
                meta = ensure_counters(meta)
                if title:
                    meta["title"] = title
                if content:
                    meta["content"] = content
                    meta["text"] = f"{title}\n{content}".strip()
                meta = apply_tags(meta, bullet_tags)
                vdb.collections["playbook"]["metadata"][internal_to_update] = meta
                vdb.persist("playbook")
                playbook_updated = True

        elif action == "add":
            # --- Add new bullet (with similarity check) ---
            query_text = f"{title}\n{content}".strip()
            if not query_text:
                print("⚠️ Skipping playbook insert: empty title/content.")
            else:
                try:
                    results = vdb.query("playbook", query_text, top_k=1)
                    if results:
                        best_match = results[0]
                        dist = best_match["distance"]
                        if dist < 0.25:
                            print(f"🔁 Found similar bullet (distance={dist:.3f}), incrementing helpful count.")
                            meta = apply_tags(best_match["metadata"], bullet_tags)
                            meta["helpful_count"] += 1
                            internal_id = best_match["internal_id"]
                            vdb.collections["playbook"]["metadata"][internal_id] = meta
                            vdb.persist("playbook")
                            playbook_updated = True
                            raise StopIteration
                except StopIteration:
                    pass
                except Exception as e:
                    print("⚠️ Similarity check failed:", e)

                if not playbook_updated:
                    try:
                        new_bid = get_next_bullet_id()
                    except Exception:
                        new_bid = f"PB{uuid.uuid4().hex[:6].upper()}"

                    payload = {
                        "text": query_text,
                        "Bullet_ID": new_bid,
                        "title": title,
                        "content": content,
                        "helpful_count": 0,
                        "harmful_count": 0,
                        "type": "playbook"
                    }
                    vdb.upsert("playbook", [payload])
                    vdb.persist("playbook")
                    playbook_updated = True

    except Exception as e:
        print("⚠️ Consolidator playbook update failed:", e)
        playbook_updated = False

    # -----------------------------
    # 5️⃣ Always update helpful/harmful counters globally
    # -----------------------------
    try:
        playbook_meta = vdb.collections["playbook"].get("metadata", {}) or {}
        for bid, tag_value in bullet_tags.items():
            bid_upper = str(bid).strip().upper()
            for internal_id, payload in playbook_meta.items():
                candidate = str(payload.get("Bullet_ID") or payload.get("bullet_id") or "").strip().upper()
                if candidate == bid_upper:
                    payload = ensure_counters(payload)
                    tag_value_lower = str(tag_value).lower()
                    if tag_value_lower == "helpful":
                        payload["helpful_count"] += 1
                    elif tag_value_lower == "harmful":
                        payload["harmful_count"] += 1
                    vdb.collections["playbook"]["metadata"][internal_id] = payload
        vdb.persist("playbook")
    except Exception as e:
        print("⚠️ Failed to globally update helpful/harmful counts:", e)

    # -----------------------------
    # 6️⃣ Update flattened state fields
    # -----------------------------
    state.consolidation_output.update({
        "playbook_update_boolean": bool(playbook_updated),
        "generator_retry_count": generator_retry_count_after,
        "route": route,
        # "decision_reason": {
        #     "score": score,
        #     "previous_retry_count": generator_retry_count_before,
        #     "action_executed": action,
        #     "playbook_updated": playbook_updated
        # }
    })

    # -----------------------------
    # 7️⃣ Mark control fields for checkpoint filtering
    # -----------------------------
    # state.visited.append("Consolidator")
    # state.current_route = "Consolidator"
    # mark iteration complete only when final route is human_feedback
    # state.is_iteration_complete = (route == "human_feedback")
    if route == "human_feedback":
        state.is_iteration_complete = False
    else:
        state.is_iteration_complete = True

    print(
        f"🧭 Consolidator Decision → Route: {route.upper()} | "
        f"Score: {score} | Retries: {generator_retry_count_after} | "
        f"Playbook Updated: {playbook_updated}"
    )

    return state


def human_feedback_node(state: LangGraphState) -> LangGraphState:
    """
    Human Feedback Node (streamlined):
      - Shows final output and reasoning
      - Takes approval and comments from the human
      - Updates state fields
      - Marks the iteration as complete
      - Persistence handled later by persist_full_task_to_vdb()
    """

    if not isinstance(state, LangGraphState):
        state = LangGraphState(**state)

    task_name = state.metadata.get("task_name", "")
    final_ans = state.generator_output.get("final_ans", "")
    reasoning = state.generator_output.get("reasoning", "")
    score = state.curator_output.get("score", 0.0)

    print("\n🧭 === HUMAN FEEDBACK NODE ===")
    print(f"Task Name: {task_name}")
    print(f"Final Answer:\n{final_ans}\n")
    print(f"Reasoning:\n{reasoning}\n")
    print(f"Curator Score: {score}")
    print("-" * 60)

    # gather human feedback interactively
    while True:
        decision = input("Approve this output? (y/n): ").strip().lower()
        if decision in ("y", "n"):
            break
        print("Please type y or n.")

    feedback_text = input("Add your comments: ").strip()
    approved = decision == "y"

    state.human_feedback_output.update({
    "human_approval_status": "Approved" if approved else "Rejected",
    "human_feedback": feedback_text
    })

    # control markers for persistence
    # state.current_route = "human_feedback"
    state.is_iteration_complete = True
    # state.metadata["is_iteration_complete"] = True

    print("✅ Human feedback recorded; iteration marked complete.")
    return state



def persist_full_task_to_vdb(graph, config, vdb, collection="states"):
    """
    Collect all completed iteration snapshots from the checkpoint history
    and persist the entire state trajectory for a single task in the vector DB.
    """

    history = list(graph.get_state_history(config))
    # print("yeh meri snapshots ki history hh",history)
    filtered_snapshots = []

    for snap in history:
        if not snap.values:
            continue
        state_values = snap.values
        # only keep snapshots that reached a "complete" iteration
        if state_values.get("is_iteration_complete", False):
            filtered_snapshots.append(state_values)
            print("LENGTH OF FILTERED SNAPSHOT LIST", len(filtered_snapshots))

    if not filtered_snapshots:
        print("⚠️ No completed iteration snapshots found — nothing persisted.")
        return

    # get task metadata
    latest_state = filtered_snapshots[-1]
    task_name = latest_state.get("metadata", {}).get("task_name", "")
    task_id = latest_state.get("task_id", str(uuid.uuid4()))

    text_parts = []
    for i, row in enumerate(filtered_snapshots, 1):
        itr = row.get("metadata", {}).get("iteration", "?")
        ans = row.get("generator_output", {}).get("final_ans", "")
        reasoning = row.get("generator_output", {}).get("reasoning", "")
        prefix = "⭐ FINAL ITERATION ⭐\n" if i == len(filtered_snapshots) else ""
        text_parts.append(f"{prefix}Iteration {itr}:\nAnswer: {ans}\nReasoning: {reasoning}\n")
    combined_text = "\n".join(text_parts)

    # assemble full task record (entire state of each iteration)
    task_record = {
        "text": combined_text,  # embedding text
        "task_id": task_id,
        "task_name": task_name,
        "total_iterations": len(filtered_snapshots),
        "iterations": filtered_snapshots,
    }

    try:
        vdb.upsert(collection, [task_record])
        vdb.persist(collection)
        print(f"✅ Saved full task '{task_name}' ({len(filtered_snapshots)} completed iterations) to '{collection}'.")
    except Exception as e:
        print(f"⚠️ Failed to persist task to VDB: {e}")


# --------------------------------------------------------------------
# -------------------- GRAPH BUILDING -------------------------------
# --------------------------------------------------------------------

def _latest_score(s):
    """Extract current score from flattened state."""
    try:
        if isinstance(s, dict):
            return float(s.get("curator_output", {}).get("score", 0.0))
        return float(s.curator_output.get("score", 0.0))
    except Exception:
        return 0.0


def _iter_len(s):
    """Extract current iteration count from flattened state metadata."""
    try:
        if isinstance(s, dict):
            return int(s.get("metadata", {}).get("iteration", 0))
        return int(s.metadata.get("iteration", 0))
    except Exception:
        return 0


def decide_next(s):
    """
    Decide next node route dynamically:
      - If score >= 9 OR iteration >= 3 → go to human_feedback
      - Else → continue generator loop
    """
    score = _latest_score(s)
    iteration = _iter_len(s)
    print(f"🧭 DECISION: score={score}, iteration={iteration}")

    if score >= 9.0 or iteration >= 3:
        return "human_feedback"
    return "generator"



# --------------------------------------------------------------------
# -------------------- GRAPH CONSTRUCTION ----------------------------
# --------------------------------------------------------------------

def build_ace_graph(vdb) -> StateGraph:
    """
    Build the full ACE LangGraph with checkpointing and access to the Vector DB.
    """
    graph = StateGraph(LangGraphState)

    # --- Node definitions with vdb injected ---
    graph.add_node("generator", lambda s: generator_node_with_playbook(s, vdb))
    graph.add_node("reflector", lambda s: reflector_node(s, vdb))
    graph.add_node("curator",   lambda s: curator_node(s, vdb))
    graph.add_node("consolidator", lambda s: consolidator_node(s, vdb))
    graph.add_node("human_feedback", lambda s: human_feedback_node(s))  # feedback node doesn’t need vdb now

    # --- Node edges ---
    graph.add_edge("generator", "reflector")
    graph.add_edge("reflector", "curator")
    graph.add_edge("curator", "consolidator")

    # Conditional routing from consolidator → generator/human_feedback
    graph.add_conditional_edges(
        "consolidator",
        decide_next,
        {"generator": "generator", "human_feedback": "human_feedback"},
    )

    # Human feedback ends the process
    graph.add_edge("human_feedback", END)

    # --- Entry point ---
    graph.set_entry_point("generator")

    return graph



# --------------------------------------------------------------------
# -------------------- MAIN EXECUTION -------------------------------
# --------------------------------------------------------------------

def main():
    """Main entrypoint for ACE agent execution with checkpointing + persistence."""

    # ✅ Initialize memory checkpointer (LangGraph’s built-in memory)
    memory = MemorySaver()

    # ✅ Initialize VDB (assumes MultiCollectionVectorDB defined elsewhere)# adjust import
    vdb = MultiCollectionVectorDB()

    # ✅ Build and compile the ACE graph
    ace_graph = build_ace_graph(vdb).compile(checkpointer=memory)

    

    # 🔹 Ask user for the task
    user_task = input("Enter your task: ").strip()
    if not user_task:
        print("No task entered — exiting.")
        return

    # 🔹 Initialize LangGraph state with dynamic user task
    initial_state = LangGraphState(generator_inputs={"user_task": user_task})

    # ✅ Config with unique thread_id for checkpoint grouping
    config = {"configurable": {"thread_id": f"ace-{uuid.uuid4()}"}}

    # 🟢 Run the ACE pipeline
    print("\n🚀 Starting ACE pipeline with checkpointing...\n")
    final_state = ace_graph.invoke(initial_state, config=config)

    # ✅ Print final state
    print("\n✅ Pipeline finished. Final state:")
    print(json.dumps(final_state, indent=2))

    # ✅ Persist full completed iteration snapshots into VDB
    print("\n💾 Persisting completed iterations to VDB...")
    persist_full_task_to_vdb(ace_graph, config, vdb)
    print("✅ Task successfully saved in VDB.")


# --------------------------------------------------------------------
# -------------------- EXECUTION ENTRY -------------------------------
# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
