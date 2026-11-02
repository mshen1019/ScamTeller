"""ScamTeller: per-utterance phone-scam early warning with a fine-tuned Qwen3-8B."""
import importlib
import os
import sys


def _import_hf_datasets() -> None:
    # The repo keeps its data in ./datasets/, which would shadow the Hugging Face `datasets`
    # library (imported by transformers.Trainer) whenever a command runs from the repo root.
    # Import the real library once with the repo root off sys.path; if it is not installed, mark
    # the name as absent so transformers' availability check does not find the data directory.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    saved = sys.path[:]
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != root]
    try:
        importlib.import_module("datasets")
    except ImportError:
        sys.modules["datasets"] = None
    finally:
        sys.path[:] = saved


_import_hf_datasets()
