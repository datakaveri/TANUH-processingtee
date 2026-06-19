"""
dep_scanner.py — scan a preprocessing script for missing dependencies and
install them using UV before the evaluation subprocess is launched.

Only third-party packages that are not already importable are installed.
stdlib modules are filtered out via sys.stdlib_module_names (Python 3.10+).
"""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

# Maps import name → PyPI package name where they differ.
# None means "skip this import entirely" (already provided by the image or not installable).
_IMPORT_TO_PACKAGE: dict[str, str | None] = {
    # imaging
    "cv2":              "opencv-python-headless",
    "PIL":              "Pillow",
    "skimage":          "scikit-image",
    # ML
    "sklearn":          "scikit-learn",
    "xgboost":          "xgboost",
    "lightgbm":         "lightgbm",
    "catboost":         "catboost",
    # data
    "bs4":              "beautifulsoup4",
    "yaml":             "PyYAML",
    "dotenv":           "python-dotenv",
    "attr":             "attrs",
    "dateutil":         "python-dateutil",
    # HuggingFace — already in image; import names differ from package names
    "sentence_transformers": "sentence-transformers",
    # Google SDKs — pre-installed; skip to avoid version conflicts
    "google":           None,
    "googleapiclient":  None,
    # already in image under a different package name
    "llama_index":      "llama-index",
    "langchain_core":   None,   # installed as part of langchain
    "langchain_community": None,
}

# Packages that are definitely in the image — skip even if import name matches
# package name 1:1, to avoid redundant install attempts.
_ALWAYS_PRESENT: frozenset[str] = frozenset({
    "torch", "torchvision", "torchaudio",
    "tensorflow", "keras",
    "jax", "jaxlib",
    "transformers", "tokenizers", "accelerate", "datasets", "huggingface_hub",
    "safetensors", "sentencepiece", "diffusers", "peft", "trl", "bitsandbytes",
    "numpy", "scipy", "pandas", "polars", "pyarrow",
    "sklearn", "scikit_learn",
    "PIL", "cv2", "skimage",
    "einops",
    "faiss", "chromadb", "qdrant_client", "pinecone",
    "openai", "anthropic", "langchain", "langgraph", "llama_index",
    "wandb", "tensorboard",
    "flask", "requests", "httpx", "tqdm", "PIL",
    "cryptography", "yaml", "dotenv",
    "onnx", "onnxruntime",
    "pydicom", "pylibjpeg",
    "sentence_transformers",
})


def _stdlib_names() -> frozenset[str]:
    if hasattr(sys, "stdlib_module_names"):
        return sys.stdlib_module_names  # Python 3.10+
    # Fallback: use sys.builtin_module_names (smaller set — acceptable fallback)
    return frozenset(sys.builtin_module_names)


def extract_imports(script_path: Path) -> set[str]:
    """Parse script_path with ast and return all top-level import root names."""
    source = script_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(script_path))
    except SyntaxError as exc:
        raise ValueError(f"Could not parse {script_path.name}: {exc}") from exc

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
            # relative imports (node.module is None) are intra-package — skip
    return names


def _is_importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def install_missing_deps(script_path: Path) -> tuple[list[str], list[str]]:
    """
    Scan script_path, determine which third-party packages are missing,
    and install them with `uv pip install --system`.

    Returns (installed, skipped) — both are lists of package specs.
    Raises RuntimeError with captured stderr if uv exits non-zero.
    """
    stdlib = _stdlib_names()
    raw_imports = extract_imports(script_path)

    to_install: list[str] = []
    skipped: list[str] = []

    for name in sorted(raw_imports):
        # stdlib
        if name in stdlib:
            skipped.append(f"{name} (stdlib)")
            continue

        # explicitly marked as skip
        if name in _IMPORT_TO_PACKAGE and _IMPORT_TO_PACKAGE[name] is None:
            skipped.append(f"{name} (provided by image)")
            continue

        # known to always be present in the image
        if name in _ALWAYS_PRESENT:
            skipped.append(f"{name} (image dep)")
            continue

        # already importable (installed)
        if _is_importable(name):
            skipped.append(f"{name} (already installed)")
            continue

        # resolve import name → package name
        package = _IMPORT_TO_PACKAGE.get(name, name)
        to_install.append(package)

    if not to_install:
        return [], skipped

    cmd = ["/bin/uv", "pip", "install", "--system"] + to_install
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"UV failed to install preprocessing dependencies "
            f"({', '.join(to_install)}):\n{result.stderr}"
        )

    return to_install, skipped
