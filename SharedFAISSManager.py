import os
import faiss
import pickle
from typing import Dict, Any, Tuple

# ✅ Cross-platform file locking setup
if os.name == "nt":  # Windows
    import portalocker

    def lock_file(file_obj):
        portalocker.lock(file_obj, portalocker.LOCK_EX)

    def unlock_file(file_obj):
        portalocker.unlock(file_obj)

else:  # Linux / macOS
    import fcntl

    def lock_file(file_obj):
        fcntl.flock(file_obj, fcntl.LOCK_EX)

    def unlock_file(file_obj):
        fcntl.flock(file_obj, fcntl.LOCK_UN)


class SharedFAISSManager:
    """
    Thread/process-safe shared FAISS manager.
    Works on both Windows and Linux using portalocker/fcntl.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _paths(self, name: str) -> Tuple[str, str, str]:
        idx = os.path.join(self.base_path, f"{name}.index")
        meta = os.path.join(self.base_path, f"{name}.pkl")
        lock = os.path.join(self.base_path, f"{name}.lock")
        return idx, meta, lock

    def _lock(self, lock_path: str):
        fd = open(lock_path, "a+")
        lock_file(fd)
        return fd

    def _unlock(self, fd):
        unlock_file(fd)
        fd.close()

    def save_vectors(self, name: str, index, metadata: Dict[int, Any]):
        idx_path, meta_path, lock_path = self._paths(name)
        lock_fd = self._lock(lock_path)
        try:
            faiss.write_index(index, idx_path)
            with open(meta_path, "wb") as f:
                pickle.dump(metadata, f)
        finally:
            self._unlock(lock_fd)

    def load_vectors(self, name: str):
        idx_path, meta_path, lock_path = self._paths(name)
        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            return None, {}
        lock_fd = self._lock(lock_path)
        try:
            index = faiss.read_index(idx_path)
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            return index, metadata
        finally:
            self._unlock(lock_fd)
