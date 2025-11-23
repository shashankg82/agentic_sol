import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, Iterable, List, Union
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
        "bullet_context": []
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
        "playbook_update_boolean": False,
        "generator_retry_count": 0,
    }

    human_feedback_output: Dict[str, Any] = {
        "human_approval_status": None,
        "human_feedback": None
    }


class LangGraphState(BaseModel):
    iterations: List[IterationSnapshot] = []
    input_task: str = ""


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



"""
faiss_multicollection_vector_db.py

Updated SimpleVectorDB -> MultiCollectionVectorDB
- Supports multiple named collections (two by default: 'playbook' and 'states')
- Uses sentence-transformers `all-MiniLM-L6-v2` for local embeddings
- Stores one FAISS index + metadata file per collection:
    vectors_{collection}.npy and metadata_{collection}.pkl
- Methods: upsert(collection, docs), query(collection, k), persist(collection=None), list_collections(), get_collection_size(collection)

Save and run as a module. Instantiate as shown at the bottom.
"""

import os
import pickle
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------- CONFIG --------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_FILE_TEMPLATE = "vectors_{name}.npy"       # file per collection
METADATA_FILE_TEMPLATE = "metadata_{name}.pkl"   # metadata per collection

DEFAULT_COLLECTIONS = ["playbook", "states"]
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

vdb = MultiCollectionVectorDB()
# -------------------- usage example --------------------

def get_path(rec: Dict[str, Any], path: Union[str, Iterable[str]], default: Any = "") -> Any:
    """
    Safe path getter for mixed/legacy records.
    - path may be 'a.b.c' or ['a','b','c']
    - It will try these fallbacks:
        1) Look for nested keys along the path (rec['a']['b']['c'])
        2) If top-level 'metadata' exists and is a dict, try metadata['a']['b']...
        3) If direct top-level key exists (rec['a.b.c'] or rec['a']), return that
    - Returns `default` if nothing found.
    """
    if rec is None or not isinstance(rec, dict):
        return default

    # normalize path to list
    if isinstance(path, str):
        parts = [p for p in path.split(".") if p]
    else:
        parts = [str(p) for p in path]

    # 1) Try nested lookup on rec
    cur = rec
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            cur = None
            break
    if cur is not None:
        return cur

    # 2) Try nested lookup inside rec['metadata'] if present
    meta = rec.get("metadata")
    if isinstance(meta, dict):
        cur = meta
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                cur = None
                break
        if cur is not None:
            return cur

    # 3) Try direct top-level keys that match the joined path (some payloads store flattened keys)
    joined = ".".join(parts)
    if joined in rec:
        return rec[joined]
    # also try first-part fallback (rec['a'])
    if parts[0] in rec:
        return rec[parts[0]]

    return default



from typing import List, Dict, Any
import json

# 1) fetch_all_from_vdb: returns list of payload dicts sorted by internal integer id
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


# 2) build_task_memory_text: unchanged but keep it here for completeness
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


# 3) _format_all_vdb_records_for_prompt: adapted to multi-collection payloads
def _format_all_vdb_records_for_prompt(records: List[Dict[str, Any]]) -> str:
    """
    Convert all records into a readable memory summary for injection into a system prompt.
    Expects each record to be a payload dict (contains 'text' and optional metadata fields).
    """
    if not records:
        return "No prior memory found."

    lines = []
    for rec in records:
        # rec typically contains fields like 'text', 'Bullet_ID', 'task_id', etc.
        internal = rec.get("_internal_id", "")
        bullet_id = rec.get("Bullet_ID") or rec.get("bullet_id", "")
        task_id = rec.get("task_id", "")
        task_name = rec.get("task_name", "")
        total_iters = rec.get("total_iterations", rec.get("iterations", ""))
        score = rec.get("score", "")
        human_status = None
        # try nested fields
        hf = rec.get("human_feedback_output") or {}
        if isinstance(hf, dict):
            human_status = hf.get("human_approval_status")

        header = f"- internal_id: {internal} | bullet_id: {bullet_id} | task_id: {task_id} | name: {task_name} | iters: {total_iters} | score: {score} | human_approved: {human_status}"
        lines.append(header)

        text = rec.get("text") or rec.get("page_content") or rec.get("Content") or rec.get("content") or ""
        if text:
            preview = text if len(text) <= 2000 else (text[:2000] + "...(truncated)")
            lines.append(preview)
        else:
            # fallback to showing metadata if no text present
            # exclude big fields if present
            meta_copy = {k: v for k, v in rec.items() if k != "text"}
            try:
                lines.append("metadata: " + json.dumps(meta_copy, default=str))
            except Exception:
                lines.append(str(meta_copy))
        lines.append("")  # blank line between records

    return "\n".join(lines)
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




# ----------------------
# Main generator node adapted to MultiCollectionVectorDB
# ----------------------
def generator_node_with_playbook(
    state: LangGraphState,
    vdb,  # instance of MultiCollectionVectorDB
    playbook_override: Optional[List[Dict[str, Any]]] = None,
    memory_collection: str = "states",
    playbook_collection: str = "playbook"
) -> LangGraphState:
    
    
    """
    Generator node which uses the multi-collection vdb.
    - memory comes from `memory_collection` (default 'states')
    - playbook from `playbook_collection`
    - updates helpful_content counts in the playbook collection for bullets actually used
    """
    # Reuse same task_id if continuing same task
    # task_id = (
    #     state.iterations[-1].task_id
    #     if state.iterations else str(uuid.uuid4())
    #     )
    # iteration_num = len(state.iterations) + 1
    # user_task = state.iterations[-1].generator_inputs.get("user_task") if state.iterations else "Define task"

    

    if not state.iterations:
        # First time generator runs for this task
        task_id = str(uuid.uuid4())
        iteration_num = 1
        user_task = getattr(state, "input_task", "")
    else:
        # Continuing the same task
        prev_iter = state.iterations[-1]
        task_id = prev_iter.task_id
        iteration_num = prev_iter.metadata.get("iteration", 0) + 1
        user_task = prev_iter.generator_inputs.get("user_task", "")

    # ------------------------
    # 🔹 Print and debug
    # ------------------------
    print(f"\n🧠 Running generator for task_id={task_id} | iteration={iteration_num}")
    print(f"Task: {user_task}")

    # fetch entire task memory from states collection
    try:
        all_records = fetch_all_from_vdb(vdb, memory_collection)
        
    except Exception:
        all_records = []
    #  memory_summary = _format_all_vdb_records_for_prompt(all_records)


    summaries = []
    for rec in all_records:
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
        print("⚠️ No prior approved tasks found — running LLM with empty memory.")
        memory_summary = "No prior approved tasks found in memory."
    else:
        lines = ["### Prior Approved Tasks and Answers ###\n"]
        for s in summaries:
            approval = "✅ Approved" if s.get("human_approval_status") else "❌ Not Approved"
            lines.append(
                f"- **Task ID:** {s.get('task_id', '')}\n"
                f"  • **Task Name:** {s.get('task_name', '')}\n"
                f"  • **User Request:** {s.get('user_task', '')}\n"
                f"  • **Final Answer:** {s.get('final_ans', '')}\n"
                f"  • **Score:** {s.get('score', '')}\n"
                f"  • **Human Feedback:** {s.get('human_feedback', '')} | {approval}\n"
            )
        memory_summary = "\n".join(lines)




    # print("YEH GENERATOR KA MEMORY SUMMARY HH", memory_summary)
    # print("-"*60)



    # fetch playbook rows
    if playbook_override is not None:
        playbook_rows = playbook_override
    else:
        try:
            playbook_rows = fetch_full_playbook_from_vdb(vdb, playbook_collection)
        except Exception:
            playbook_rows = []

    playbook_text = format_playbook_for_prompt(playbook_rows)

    # print("YEH GENERATOR KA PLAYBOOK FETCH HH", playbook_text)
    # print("-"*60)

    # build system prompt (same structure you provided)
    system_prompt = """
You are the Generator Agent specializing in producing precise and robust Excel formulas based on research task descriptions.

Your primary objectives:
- Strict adherence to existing playbook rules.
- Incorporation of relevant past experiences from memory.
- Ensure generated formulas are syntactically valid and logically sound.
- Minimize revision iterations and maximize formula acceptance rates.

You have access to:
- Playbook rules: {playbook_text}
- Past memory of formulas created by generator: {memory_summary}

Return your output with EXACTLY these labeled sections (UPPERCASE labels only):

FINAL_ANS:
- Provide Excel formula(s) with column names or cell references. Ready to paste.

REASONING:
- Explain why the formula(s) work and any assumptions/edge-cases.

BULLETS_USED:
- Summarize which bullets influenced your reasoning (short explanation).

BULLET_CONTEXT:
- Output a JSON-like list of lists — each containing BULLET_ID, TITLE, CONTENT — e.g.
  [[PB0024, "Calculate Rolling Averages with OFFSET", "Use =AVERAGE(...) to calculate..."],
   [PB0025, "Handle Missing Data", "Explain how missing values affect rolling returns"]]

Instructions:
- Use the playbook rules as guiding principles.
- Integrate insights from previous feedback to avoid repeating mistakes.
- Generate the most accurate and efficient Excel formula possible.
"""
    print("dvbdhvgdhvskhdvb", system_prompt)
    
    user_prompt = f"Task: {user_task}"
    raw_output = call_llm(system_prompt, user_prompt)


    

    # --- parse sections ---
    normalized = (raw_output or "").replace("\r\n", "\n")

    print("YEH GENERATOR LLM KA OUTPUT HH:", normalized)
    print("-" * 90)

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

    labels = ["FINAL_ANS:", "REASONING:", "BULLETS:", "BULLET_CONTEXT:"]
    final_ans_text = extract_section_by_labels(normalized, "FINAL_ANS:", labels[1:])
    reasoning_text = extract_section_by_labels(normalized, "REASONING:", labels[2:])
    bullets_text = extract_section_by_labels(normalized, "BULLETS:", ["BULLET_CONTEXT:", "FINAL_ANS:", "REASONING:"])
    bullet_context_text = extract_section_by_labels(normalized, "BULLET_CONTEXT:", [])

    # --- parse BULLET_CONTEXT robustly into list of {"id","title","content"} ---
    def parse_bullet_context(text: str):
        txt = (text or "").strip()
        if not txt or txt.upper() == "NONE":
            return []

        # 1) try strict JSON
        try:
            parsed = json.loads(txt)
            out = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, list) and len(item) >= 1:
                        bid = str(item[0]).strip()
                        title = str(item[1]).strip() if len(item) > 1 else ""
                        content = str(item[2]).strip() if len(item) > 2 else ""
                        out.append({"id": bid, "title": title, "content": content})
                if out:
                    return out
        except Exception:
            pass

        # 2) attempt light coercion (single->double quotes, quote B-IDs)
        try:
            cand = txt.replace("'", '"')
            cand = re.sub(r'\[([^\[\]]*?)\]', lambda m: "[" + re.sub(r'(\bB-\d+\b)', r'"\1"', m.group(1)) + "]", cand)
            parsed = json.loads(cand)
            out = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, list) and len(item) >= 1:
                        bid = str(item[0]).strip()
                        title = str(item[1]).strip() if len(item) > 1 else ""
                        content = str(item[2]).strip() if len(item) > 2 else ""
                        out.append({"id": bid, "title": title, "content": content})
                if out:
                    return out
        except Exception:
            pass

        # 3) regex: extract bracketed rows like [B-101, title101, content101]
        rows = re.findall(r'\[([^\[\]]+?)\]', txt)
        parsed_rows = []
        for r in rows:
            parts = [p.strip() for p in re.split(r'\s*,\s*', r)]
            if len(parts) >= 1:
                bid = parts[0]
                title = parts[1] if len(parts) > 1 else ""
                content = parts[2] if len(parts) > 2 else ""
                parsed_rows.append({"id": bid, "title": title, "content": content})
        if parsed_rows:
            return parsed_rows

        # 4) fallback: extract bare B-IDs
        ids = re.findall(r'(B-\d+)', txt, flags=re.IGNORECASE)
        return [{"id": i.upper(), "title": "", "content": ""} for i in ids]

    parsed_bullet_context = parse_bullet_context(bullet_context_text)


    # print("Yeh GENERATOR KA BUULET_CONTEXT HH:", parsed_bullet_context)
    # print("-" * 60)

    # build bullet id list (simple list) for backward compatibility
    parsed_bullet_ids = [entry["id"].upper() for entry in parsed_bullet_context if entry.get("id")]

    # fallback: if none found, attempt to find ids in bullets_text
    if not parsed_bullet_ids and bullets_text:
        ids_from_bullets = re.findall(r'(B-\d+)', bullets_text, flags=re.IGNORECASE)
        if ids_from_bullets:
            parsed_bullet_ids = [i.upper() for i in ids_from_bullets]
            parsed_bullet_context = [{"id": i.upper(), "title": "", "content": ""} for i in ids_from_bullets]

    # deduplicate while preserving order and normalize
    seen = set()
    unique_ids = []
    unique_context = []
    for entry in parsed_bullet_context:
        bid = (entry.get("id") or "").strip().upper()
        if not bid or bid in seen:
            continue
        seen.add(bid)
        unique_ids.append(bid)
        unique_context.append({"id": bid, "title": (entry.get("title") or "").strip(), "content": (entry.get("content") or "").strip()})

    # ensure we have at least an empty context list if ids exist but no context detail
    if not unique_context and parsed_bullet_ids:
        unique_ids = []
        for bid in parsed_bullet_ids:
            ub = bid.strip().upper()
            if ub not in seen:
                seen.add(ub)
                unique_ids.append(ub)
        unique_context = [{"id": bid, "title": "", "content": ""} for bid in unique_ids]

    # --- create iteration and append to state ---
    iteration = IterationSnapshot(
        task_id=task_id,
        metadata={"task_name": user_task, "iteration": iteration_num, "score": 0.0},
        generator_inputs={"user_task": user_task},
        generator_output={
            "final_ans": final_ans_text,
            "reasoning": reasoning_text,
            "bullet_context": unique_context
        }
    )
    state.iterations.append(iteration)

    # return updated state
    return state


import json
from typing import List, Dict, Any

def reflector_node(state: "LangGraphState", vdb) -> "LangGraphState":
    """
    Reflector node adapted to the MultiCollectionVectorDB and safe helper `fetch_all_from_vdb`.
    - Reads prior examples from the 'states' collection using fetch_all_from_vdb(vdb, "states")
    - Analyzes the most recent generator output (last iteration)
    - Calls the LLM (via call_llm) with a strict JSON schema request
    - Parses JSON robustly (with fallback extraction)
    - Updates iteration.reflector_output with normalized fields
    """

    iteration = state.iterations[-1]

    # -----------------------------
    # Present context
    # -----------------------------
    gen_final = iteration.generator_output.get("final_ans", "")
    gen_reasoning = iteration.generator_output.get("reasoning", "")
    gen_bullets = iteration.generator_output.get("bullet_context", []) or []
    user_task = iteration.generator_inputs.get("user_task", "")

    # -----------------------------
    # Build full history from vectordb (use safe helper)
    # -----------------------------
    history_items: List[Dict[str, Any]] = []
    try:
        all_rows = fetch_all_from_vdb(vdb, "states")
    except Exception:
        all_rows = []


    summaries = []
    for rec in all_rows:
        task_id = rec.get("task_id", "")
        for itr in rec.get("iterations", []):
            meta = itr.get("metadata", {})
            gen_out = itr.get("generator_output", {})
            refl_out = itr.get("reflector_output", {})
            human_fb = itr.get("human_feedback_output", {})

            summaries.append({
                "task_id": task_id,
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


    

    if not summaries:
        print("⚠️ No prior reflector insights found in memory — running LLM with empty context.")
        memory_summary = (
            "⚠️ No prior reflector insights found in memory.\n"
            "Use only the generator’s current output for analysis.\n"
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
                f"  • **Root Cause Analysis:** {s['reflector_rootcauseanalysis']}\n"
                f"  • **Correct Approach:** {s['reflector_correctapproach']}\n"
                f"  • **Key Insights:** {s['reflector_keyinsights']}\n"
                f"  • **Approval:** {approval}\n"
            )
        memory_summary = "\n".join(lines)



    print("YEH REFLECTOR KA MEMORY SUMMARY HH", memory_summary)
    print("-"*60)


    # # Normalize stored payloads into a compact schema expected by the prompt
    # def get_path(rec, path, default=None):
    # # check normal nested keys
    #     keys = path.split(".")
    #     cur = rec
    #     for k in keys:
    #         if not isinstance(cur, dict):
    #             return default
    #         cur = cur.get(k)
    #         if cur is None:
    #             break

    #     if cur is not None:
    #         return cur
        
    #     if path in rec:
    #         return rec[path]
    #     return default

    # for rec in all_rows:
    #     history_items.append({
    #         "task_id": get_path(rec, "task_id", ""),
    #         "metadata.task_name": get_path(rec, "metadata.task_name", ""),
    #         # "generators_input_user_task": get_path(rec, "generator_inputs.user_task", ""),
    #         "generators_output_final_ans": get_path(rec, "generator_output.final_ans", ""),
    #         "generators_output_reasoning": get_path(rec, "generator_output.reasoning", ""),
    #         "generators_output_bulletlist": get_path(rec, "generator_output.bulletlist", []) or [],
    #         "human_feedback_output_human_approval_status": get_path(rec, "human_feedback_output.human_approval_status", None),
    #         "reflector_output_reasoning": get_path(rec, "reflector_output.reasoning", ""),
    #         "reflector_output_error_identification": get_path(rec, "reflector_output.error_identification", []),
    #         "reflector_output_keyinsights": get_path(rec, "reflector_output.keyinsights", []),
    #         "reflector_output_correctapproach": get_path(rec, "reflector_output.correctapproach", ""),
    #     })


    # -----------------------------
    # Build compact previews for prompt (avoid huge prompt by truncating fields)
    # -----------------------------
    # def preview_hist_item(it: Dict[str, Any], max_chars: int = 800) -> str:
    #     preview = {
    #         "task_id": it.get("task_id"),
    #         "metadata.task_name": it.get("metadata.task_name"),
    #         # "generators_input_user_task": (it.get("generators_input_user_task") or "")[:max_chars],
    #         "generators_output_final_ans": (it.get("generators_output_final_ans") or "")[:max_chars],
    #         "generators_output_reasoning": (it.get("generators_output_reasoning") or "")[:max_chars],
    #         "generators_output_bulletlist": it.get("generators_output_bulletlist", []),
    #         "human_feedback": it.get("human_feedback_output_human_approval_status"),
    #         "reflector_keyinsights": it.get("reflector_output_keyinsights")
    #     }
    #     return json.dumps(preview, ensure_ascii=False)

    # history_block = "\n".join(preview_hist_item(h) for h in history_items)
    # if not history_block:
    #     history_block = "No prior history available."


    # print("YEH REFLECTOE KA INPUT HH", history_block)
    # print("-"*60)

    # # -----------------------------
    # Prepare the LLM prompt with schema instructions
    # -----------------------------
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

    system_prompt = """
You are the Reviewer Agent tasked with evaluating the correctness, robustness, and optimization of Excel formulas generated for research tasks.

Your goals:
- Provide consistent, actionable, and precise feedback aligned with known best practices.
- Leverage historical reviews and playbook guidelines to maintain evaluation consistency.
- Suggest meaningful updates to the playbook when new gaps or optimizations are identified.
- Maximize feedback acceptance rate while minimizing contradictory or redundant comments.

You have access to:
- Current formula to review: {gen_final}
- Task context: {user_task}
- Relevant playbook rules: {json.dumps(gen_bullets, ensure_ascii=False)}
- Prior reviewer feedback and evaluations on similar formulas: {memory_summary}

Evaluate the given formula based on:
- Accuracy and logical correctness
- Robustness and error handling
- Compliance with domain-specific heuristics and past reviewer comments

Provide:
- Detailed, constructive feedback or approval
- Suggestions for new or updated playbook rules if needed

Return structured feedback ONLY in the required JSON format:
{{
  "reasoning": "[Your chain of thought / reasoning / detailed analysis and calculations]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "rootcauseanalysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "keyinsights": "[What strategy, formula, or principle should be remembered to avoid this error?]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}
"""


    user_prompt = (
        "Present generator output (primary context):\n"
        f"FINAL_ANS:\n{gen_final}\n\n"
        f"REASONING:\n{gen_reasoning}\n\n"
        f"BULLET_CONTEXT:\n{json.dumps(gen_bullets, ensure_ascii=False)}\n\n"
        f"TASK INPUT:\n{user_task}\n\n"
        "Analyze and return JSON as instructed in the system message."
    )

    # -----------------------------
    # Call LLM and parse JSON robustly (with fallbacks)
    # -----------------------------
    raw_response = call_llm(system_prompt, user_prompt)


    print("YEH REFLECTOR KA LLM RESPOND HH: ", raw_response)
    print("-" * 90)

    parsed = None
    try:
        parsed = json.loads(raw_response)
    except Exception:
        # Try to extract the first JSON object-looking substring
        try:
            start = raw_response.index("{")
            end = raw_response.rindex("}") + 1
            parsed = json.loads(raw_response[start:end])
        except Exception:
            parsed = None

    # -----------------------------
    # Handle parsing failure (fallback)
    # -----------------------------
    if not isinstance(parsed, dict):
        # gen_bullets is expected to be a list-of-lists: [[id, title, content], ...]
        normalized_ids = []
        for item in gen_bullets or []:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                first = item[0]
                if isinstance(first, str):
                    normalized_ids.append(first.strip().upper())

        # Deduplicate while preserving order
        seen = set()
        gen_ids = []
        for gid in normalized_ids:
            if gid and gid not in seen:
                seen.add(gid)
                gen_ids.append(gid)

        iteration.reflector_output.update({
            "reasoning": raw_response,
            "error_identification": ["analysis_not_parsed_from_llm"],
            "rootcauseanalysis": "",
            "correctapproach": "",
            "keyinsights": [],
            "bullet_tags": {b: "harmful" for b in gen_ids}  # Conservative default
        })
        return state

    # -----------------------------
    # If valid JSON parsed, update the reflector output
    # -----------------------------
    reflector_output = parsed.get("reflector_output", {})
    iteration.reflector_output.update({
        "reasoning": reflector_output.get("reasoning", ""),
        "error_identification": reflector_output.get("error_identification", []),
        "rootcauseanalysis": reflector_output.get("rootcauseanalysis", ""),
        "correctapproach": reflector_output.get("correctapproach", ""),
        "keyinsights": reflector_output.get("keyinsights", []),
        "bullet_tags": reflector_output.get("bullet_tags", {}),
    })

    return state

    



def curator_node(state: LangGraphState, vdb) -> LangGraphState:
    """
    Curator adapted to MultiCollectionVectorDB.
    - Reads playbook from vdb.collections['playbook']
    - May add or update entries in that collection
    """
    iteration = state.iterations[-1]

    # Present context
    present_task_name = iteration.metadata.get("task_name", "")
    gen_final_ans = iteration.generator_output.get("final_ans", "")
    refl_reasoning = iteration.reflector_output.get("reasoning", "")
    refl_errors = iteration.reflector_output.get("error_identification", [])
    refl_rootcause = iteration.reflector_output.get("rootcauseanalysis", "")
    refl_correct = iteration.reflector_output.get("correctapproach", "")
    refl_keyinsights = iteration.reflector_output.get("keyinsights", [])


    # -----------------------
    # Build memory from 'states' collection using same logic as reflector_node
    # -----------------------
    history_items = []
    try:
        all_rows = fetch_all_from_vdb(vdb, "states")
    except Exception:
        all_rows = []

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
        print("⚠️ No prior curator insights found in memory — running LLM with empty context.")
        memory_summary = (
            "⚠️ No prior curator insights found in memory.\n"
            "This is the first task being processed.\n"
            "Use only the current generator and reflector outputs for your reasoning.\n"
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


    # for rec in all_rows:
    #     # use get_path (safe path getter) for nested fields
    #     try:
    #         history_items.append({
    #             "task_id": get_path(rec, "task_id", ""),
    #             "metadata.task_name": get_path(rec, "metadata.task_name", ""),
    #             "metadata.score": get_path(rec, "metadata.score", get_path(rec, "score", None)),
    #             "human_feedback_output.human_approval_status": get_path(rec, "human_feedback_output.human_approval_status", None),
    #             "curator_output.action": get_path(rec, "curator_output.action", ""),
    #             "curator_output.content": get_path(rec, "curator_output.content", "")
    #         })
    #     except Exception:
    #         # skip malformed record
    #         continue

    # # Build compact previews (truncate long content)
    # def preview_mem_item(it: Dict[str, Any]) -> str:
    #     score = it.get("metadata.score")
    #     try:
    #         score_repr = f"{float(score)}" if score is not None else "None"
    #     except Exception:
    #         score_repr = str(score)
    #     human = it.get("human_feedback_output.human_approval_status")
    #     human_repr = "approved" if human is True else ("rejected" if human is False else "unknown")
    #     tid = it.get("task_id") or ""
    #     tname = (it.get("metadata.task_name") or "")[:120]
    #     action = it.get("curator_output.action") or ""
    #     content = (it.get("curator_output.content") or "")
    #     if isinstance(content, (list, dict)):
    #         content = json.dumps(content, ensure_ascii=False)
    #     content_preview = content
    #     return json.dumps({
    #         "task_id": tid,
    #         "task_name": tname,
    #         "score": score_repr,
    #         "human": human_repr,
    #         "action": action,
    #         "content_preview": content_preview
    #     }, ensure_ascii=False)

    # memory_block = "\n".join(preview_mem_item(h) for h in history_items) or "No prior memory available."


    # Fetch playbook rows from vdb (use the helper for normalization)
    try:
        playbook_rows = fetch_full_playbook_from_vdb(vdb, "playbook") if (vdb is not None and hasattr(vdb, "collections")) else []
    except Exception:
        playbook_rows = []

    playbook_text = format_playbook_for_prompt(playbook_rows) if playbook_rows else "No playbook bullets found."


    print("YEH CURATOR KA MEMORY SUMMARY HH", memory_summary)
    print("-"*60)
    print("YEH CURATOR KA PLAYBOOK SE INPUT HH", playbook_text)
    print("-"*60)

    # LLM instructions (same schema you defined)
    schema_instructions = (
        "Return ONLY a valid JSON object (no extra commentary) using the exact structure below.\n\n"
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

    system_prompt = (
    """
You are the Curator Agent responsible for validating and evolving the playbook of Excel formula generation rules based on reviewer feedback and system history. You act as a scorer for the generator's work.

Your objectives:
- Ensure all playbook updates are consistent, beneficial, and non-redundant.
- Leverage historical curator decisions and playbook versions to resolve conflicts.
- Maintain a balance between playbook growth and pruning to optimize quality.
- Minimize unnecessary escalation to human approval without compromising safety.

Available inputs:
- Reviewer feedback and proposed playbook updates:
"Task name: {present_task_name}\\n\\n"
Generator FINAL_ANS:
{gen_final_ans}
"Reflector reasoning:
{refl_reasoning}\\n\\n"
"Reflector error_identification:
{json.dumps(refl_errors, ensure_ascii=False)}\\n\\n"
"Reflector rootcauseanalysis:
{refl_rootcause}\\n\\n"
"Reflector correctapproach:
{refl_correct}\\n\\n"
"Reflector keyinsights (candidate learning points):
{json.dumps(refl_keyinsights, ensure_ascii=False)}\\n\\n"

- Current playbook: {playbook_text}
- Historical curator decisions on similar feedback: {memory_summary}

Analyze and act according to:
- Consistency with existing playbook rules and previous decisions
- Effectiveness and acceptance of similar past updates
- Conflicts or redundancies in suggested changes
- Overall feasibility and quality of proposed updates

Proceed to:
- Update the playbook appropriately (confirm, modify, or reject updates)
- Decide if the formula and updates should proceed for human approval or return to Generator for refinement

Return ONLY a valid JSON object with this exact structure:
{
  "curator_output": {
      "action": "add" | "update",
      "title": string|null,
      "content": string|null,
      "bullet_id": string|null,
      "score": number    # 1-10 confidence/quality rating for generator output
  }
}
"""
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
        "Using the full playbook and the memory above, decide whether the candidate insight should be added, used to refine an existing bullet (update), or ignored.\n"
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

    # Debug
    print("CURATOR LLM KA OUTPUT:", parsed)
    print("-" * 90)

    # Fallback if parsing failed: use heuristic
    

    curator_block = parsed.get("curator_output", {}) or {}
    final_action = curator_block.get("action", "ignore")
    title = curator_block.get("title")
    content = curator_block.get("content")
    bullet_id = curator_block.get("bullet_id")
    score = curator_block.get("score", 0)

# # optional: convert to list for consistency
# if isinstance(final_action, str):
#     final_action = [final_action]


    iteration.curator_output.update({
        "action": [final_action],
        "content": content,
        "title": title,
        "bullet_id": bullet_id,
         "score": score
    })
    iteration.metadata["score"] = score
        # no DB changes in fallback (you can change to auto-add if desired)
    return state

def get_next_bullet_id(prefix="PB", tracker_path="playbook_bullet.json"):
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
    Consolidator Node (updated version):
    - Decides routing:
        * If curator score >= 9 → human_feedback
        * If curator score < 9 → retry generator up to 3 times, then human_feedback
    - Smart playbook management:
        * Before adding a new bullet, checks FAISS for semantic similarity.
        * If a similar bullet exists, increments helpful/harmful counts instead of creating duplicate.
    """
    iteration = state.iterations[-1]

    # --- Extract curator outputs ---
    curator_action = iteration.curator_output.get("action", []) or []
    if isinstance(curator_action, str):
        curator_action = [curator_action]
    curator_action_lc = [a.strip().lower() for a in curator_action]
    action = curator_action_lc[0] if curator_action_lc else "ignore"

    title = (iteration.curator_output.get("title") or "").strip()
    content = (iteration.curator_output.get("content") or "").strip()
    bullet_id = iteration.curator_output.get("bullet_id") or None
    bullet_tags = iteration.reflector_output.get("bullet_tags", {}) or {}

    # --- Score and retry logic ---
    try:
        score = float(iteration.curator_output.get("score", 0.0))
    except Exception:
        score = 0.0

    task_id = getattr(iteration, "task_id", None) or iteration.metadata.get("task_id")
    generator_retry_count_before = sum(
        1 for it in state.iterations
        if getattr(it, "task_id", None) == task_id and
           it.consolidation_output.get("route", "") == "generator"
    )

    max_generator_rounds = 3
    route = "ignore"
    generator_retry_count_after = generator_retry_count_before

    if score >= 9.0:
        route = "human_feedback"
    else:
        if generator_retry_count_before < max_generator_rounds:
            route = "generator"
            generator_retry_count_after += 1
        else:
            route = "human_feedback"

    playbook_updated = False

    # --- Helper functions ---
    def ensure_counters(meta: Dict[str, Any]):
        if not isinstance(meta, dict):
            return
        meta.setdefault("helpful_count", int(meta.get("helpful_content", 0)))
        meta.setdefault("harmful_count", int(meta.get("harmful_content", 0)))
        return meta

    def apply_tags(meta: Dict[str, Any], tags: Dict[str, Any]):
        ensure_counters(meta)
        for _, v in (tags or {}).items():
            if str(v).lower() == "helpful":
                meta["helpful_count"] += 1
            elif str(v).lower() == "harmful":
                meta["harmful_count"] += 1
        return meta

    # --- Smart Playbook Handling ---
    # --- Smart Playbook Handling ---
    try:
        # Count helpful/harmful signals from reflector bullet_tags (if any)
        helpful_count = sum(1 for t in (bullet_tags or {}).values() if str(t).lower() == "helpful")
        harmful_count = sum(1 for t in (bullet_tags or {}).values() if str(t).lower() == "harmful")

        if action in ("ignore", ""):
            playbook_updated = False

        elif action == "update":
            # --- Update existing record ---
            if bullet_id:
                playbook_meta = vdb.collections["playbook"].get("metadata", {}) or {}
                internal_to_update = None
                bid_upper = bullet_id.strip().upper()

                for internal_id, payload in playbook_meta.items():
                    candidate = str(payload.get("Bullet_ID") or payload.get("bullet_id") or "").strip().upper()
                    if candidate == bid_upper:
                        internal_to_update = internal_id
                        break

                if internal_to_update is not None:
                    meta = playbook_meta[internal_to_update]
                    if isinstance(meta.get("metadata"), dict):
                        meta = meta["metadata"]

                    if title:
                        meta["title"] = title
                    if content:
                        meta["content"] = content
                        meta["text"] = f"{title}\n{content}".strip()

                    meta = apply_tags(meta, bullet_tags)
                    vdb.collections["playbook"]["metadata"][internal_to_update] = meta
                    vdb.persist("playbook")
                    playbook_updated = True
                else:
                    playbook_updated = False
            else:
                playbook_updated = False

        elif action == "add":
            query_text = f"{title}\n{content}".strip()
            if not query_text:
                print("⚠️ Skipping playbook insert: Empty title/content.")
            else:
                # --- Check for semantic similarity ---
                try:
                    results = vdb.query("playbook", query_text, top_k=1)
                    if results:
                        best_match = results[0]
                        dist = best_match["distance"]
                        meta = best_match["metadata"]

                        if dist < 0.25:  # semantic threshold
                            print(f"🔁 Found similar bullet (distance={dist:.3f}), incrementing helpful count.")
                            meta = apply_tags(meta, bullet_tags)
                            meta["helpful_count"] = meta.get("helpful_count", 0) + 1

                            internal_id = best_match["internal_id"]
                            vdb.collections["playbook"]["metadata"][internal_id] = meta
                            vdb.persist("playbook")
                            playbook_updated = True
                            route = "human_feedback"
                            raise StopIteration  # stop add process
                except StopIteration:
                    pass
                except Exception as e:
                    print("⚠️ Similarity check failed:", e)

                if not playbook_updated:
                    # --- Add new entry ---
                    try:
                        new_bid = get_next_bullet_id()
                    except Exception:
                        new_bid = f"B-{uuid.uuid4().hex[:8].upper()}"

                    payload = {
                        "text": query_text,
                        "Bullet_ID": new_bid,
                        "title": title,
                        "content": content,
                        "helpful_count": helpful_count,
                        "harmful_count": harmful_count,
                        "type": "playbook"
                    }
                    vdb.upsert("playbook", [payload])
                    vdb.persist("playbook")
                    playbook_updated = True

        else:
            playbook_updated = False

    except Exception as e:
        print("⚠️ Consolidator playbook operation failed:", e)
        playbook_updated = False


    # ✅ Always increment helpful/harmful counts for bullets referenced in reflector
    try:
        if bullet_tags:
            playbook_meta = vdb.collections["playbook"].get("metadata", {}) or {}

            for bid, tag_value in bullet_tags.items():
                bid_upper = str(bid).strip().upper()
                for internal_id, payload in playbook_meta.items():
                    candidate = str(payload.get("Bullet_ID") or payload.get("bullet_id") or "").strip().upper()
                    if candidate == bid_upper:
                        if isinstance(payload.get("metadata"), dict):
                            payload = payload["metadata"]

                        tag_value_lower = str(tag_value).lower()
                        if tag_value_lower == "helpful":
                            payload["helpful_count"] = payload.get("helpful_count", 0) + 1
                        elif tag_value_lower == "harmful":
                            payload["harmful_count"] = payload.get("harmful_count", 0) + 1

                        vdb.collections["playbook"]["metadata"][internal_id] = payload

            vdb.persist("playbook")
    except Exception as e:
        print("⚠️ Failed to update helpful/harmful counts globally:", e)


    # --- Finalize state ---
    iteration.consolidation_output["playbook_update_boolean"] = bool(playbook_updated)
    iteration.consolidation_output["route"] = route
    iteration.consolidation_output["generator_retry_count"] = generator_retry_count_after
    iteration.consolidation_output.setdefault("decision_reason", {})
    iteration.consolidation_output["decision_reason"].update({
        "score": score,
        "previous_generator_retry_count": generator_retry_count_before,
        "action_executed": action,
        "playbook_updated": playbook_updated
    })

    print(
        f"🧭 Consolidator Decision → Route: {route.upper()} | "
        f"Score: {score} | Retries: {generator_retry_count_after} | "
        f"Playbook Updated: {playbook_updated}"
    )

    return state



def human_feedback_node(state: LangGraphState, vdb: MultiCollectionVectorDB) -> LangGraphState:



    """
    Human Feedback Node that:
      - Gets approval or rejection from human
      - Logs feedback
      - Saves the *entire task* (all iterations) as ONE record in vector DB
        in the form: { text: full_text, iterations: [itr1, itr2, itr3, ...], ... }
    """
    if not isinstance(state, LangGraphState):
        state = LangGraphState(**state)

    # Get latest iteration for context
    iteration = state.iterations[-1]
    task_id = iteration.task_id
    task_name = iteration.metadata.get("task_name", "")
    final_ans = iteration.generator_output.get("final_ans", "")
    reasoning = iteration.generator_output.get("reasoning", "")
    score = iteration.curator_output.get("score", 0.0)

    print("\n🧭 === HUMAN FEEDBACK NODE ===")
    print(f"Task Name: {task_name}")
    print(f"Final Answer:\n{final_ans}\n")
    print(f"Reasoning:\n{reasoning}\n")
    print(f"Curator Score: {score}")
    print("-" * 60)

    # Ask human
    while True:
        decision = input("Approve this output? (y/n): ").strip().lower()
        if decision in ("y", "n"):
            break
        print("Please type y or n.")

    feedback_text = input("Add your comments: ").strip()
    approved = decision == "y"

    # Update latest iteration with feedback
    iteration.human_feedback_output.update({
        "human_approval_status": "Approved." if approved else "Rejected.",
        "human_feedback": feedback_text 
    })


    # --- Build full combined text across all iterations ---
    combined_text = build_task_memory_text(state)

    # --- Create ONE record for the entire task ---
    task_record = {
        "text": combined_text,                          # embedding content
        "task_id": task_id,              # unique id for this task
        "total_iterations": len(state.iterations),
        "iterations": [itr.dict() for itr in state.iterations],  # full structured list
        "latest_iteration_score": score,
        "latest_human_feedback": iteration.human_feedback_output,
    }

    try:
        # ✅ Upsert exactly ONE row (whole task)
        vdb.upsert("states", [task_record])
        vdb.persist("states")
        print(f"✅ Saved complete task ({len(state.iterations)} iterations) as ONE record in 'states'.")
    except Exception as e:
        print(f"⚠️ Failed to persist task: {e}")

    return state



# -------------------- LANGGRAPH FLOW --------------------
# --- helper readers ---
def _latest_score(s):
    it = s["iterations"][-1] if isinstance(s, dict) else s.iterations[-1]
    return float(it["curator_output"]["score"] if isinstance(it, dict) else it.curator_output.get("score", 0) or 0)

def _iter_len(s):
    return len(s["iterations"]) if isinstance(s, dict) else len(s.iterations)

def decide_next(s):
    try:
        return "human_feedback" if (_latest_score(s) >= 9 or _iter_len(s) >= 3) else "generator"
    except Exception:
        return "human_feedback"

def build_ace_graph() -> StateGraph:
    graph = StateGraph(LangGraphState)

    graph.add_node("generator", lambda s: generator_node_with_playbook(s, vdb))
    graph.add_node("reflector", lambda s: reflector_node(s, vdb))
    graph.add_node("curator",   lambda s: curator_node(s, vdb))
    graph.add_node("consolidator", lambda s: consolidator_node(s, vdb))
    graph.add_node("human_feedback", lambda s: human_feedback_node(s, vdb))  # ✅ pass vdb

    graph.add_edge("generator", "reflector")
    graph.add_edge("reflector", "curator")
    graph.add_edge("curator", "consolidator")

    # ✅ map labels to nodes, not END
    graph.add_conditional_edges(
        "consolidator",
        decide_next,
        {"human_feedback": "human_feedback", "generator": "generator"},
    )

    # ✅ only end after the node actually runs
    graph.add_edge("human_feedback", END)

    graph.set_entry_point("generator")
    return graph




# -------------------- MAIN --------------------
# def main():
#     # Compile your LangGraph
#     ace_graph = build_ace_graph().compile()

#     # Define the user task
#     user_task = "Test consistency of a strategy using rolling returns."

#     # ✅ Start with an empty LangGraphState (no dummy iteration)
#     initial_state = LangGraphState(iterations=[])

#     # ✅ Create the first iteration snapshot directly with generator_inputs
#     #    This is safe because it's the starting point for the generator node.
#     first_iteration = IterationSnapshot(
#         generator_inputs={"user_task": user_task},
#         metadata={"iteration": 1, "task_name": user_task, "score": 0.0}
#     )

#     # Add that first iteration to the state
#     initial_state.iterations.append(first_iteration)

#     # 🟢 Run the ACE pipeline
#     final_state = ace_graph.invoke(initial_state)

    

# if __name__ == "__main__":
#     main()


def main():
    # Build the compiled ACE LangGraph
    ace_graph = build_ace_graph().compile()

    # 🔹 Ask user for the task dynamically
    user_task = input("Enter your task: ").strip()
    if not user_task:
        print("No task entered — exiting.")
        return
    
    
    # 🔹 Initialize empty LangGraphState
    initial_state = LangGraphState(iterations=[])
    # Store user task for generator_node to use later
    initial_state.input_task = user_task  

    # 🟢 Run the ACE pipeline
    ace_graph.invoke(initial_state)

    print("\n✅ Pipeline finished. Final state:")
    # print(final_state)


if __name__ == "__main__":
    main()