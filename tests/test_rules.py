"""AST/tokenize enforcement over src/agent_listening. Each rule has a negative sample."""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "agent_listening"


def _modules() -> dict[str, ast.Module]:
    return {p.name: ast.parse(p.read_text()) for p in sorted(SRC.glob("*.py"))}


def _imported_names(tree: ast.AST) -> list[tuple[str, int]]:
    """Resolved top-level names including `from agent_listening import gemini`."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out += [(f"{node.module}.{alias.name}" if node.module else alias.name, node.lineno)]
            if node.module:
                out.append((node.module, node.lineno))
    return out


def test_only_gemini_imports_subprocess():
    for name, tree in _modules().items():
        for mod, line in _imported_names(tree):
            top = mod.split(".")[0]
            if top == "subprocess" and name != "gemini.py":
                raise AssertionError(f"{name}:{line} imports subprocess (only gemini.py may)")

    bad = ast.parse("import subprocess\n")
    found = [mod for mod, _ in _imported_names(bad) if mod.split(".")[0] == "subprocess"]
    assert found, "checker must flag subprocess"


def test_only_tasks_imports_gemini_module():
    for name, tree in _modules().items():
        for mod, line in _imported_names(tree):
            if mod == "agent_listening.gemini" and name != "tasks.py":
                raise AssertionError(f"{name}:{line} imports agent_listening.gemini (only tasks.py may)")

    bad = ast.parse("from agent_listening import gemini\n")
    found = [mod for mod, _ in _imported_names(bad) if mod == "agent_listening.gemini"]
    assert found, "checker must flag agent_listening.gemini"


BANNED = frozenset(
    {
        "torch",
        "tensorflow",
        "essentia",
        "madmom",
        "librosa",
        "demucs",
        "demucs_infer",
        "basic_pitch",
        "allin1",
        "all_in_one",
        "jams",
        "src",
    }
)


def test_no_banned_imports():
    for name, tree in _modules().items():
        for mod, line in _imported_names(tree):
            top = mod.lstrip(".").split(".")[0]
            if top in BANNED:
                raise AssertionError(f"{name}:{line} imports banned module {top}")

    bad = ast.parse("import librosa\n")
    found = [mod for mod, _ in _imported_names(bad) if mod.lstrip(".").split(".")[0] in BANNED]
    assert found, "checker must flag banned imports"


def _top_level_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """Imports executed eagerly at module import time (top-level statements only)."""
    out = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out += [(f"{node.module}.{alias.name}" if node.module else alias.name, node.lineno)]
            if node.module:
                out.append((node.module, node.lineno))
    return out


def test_cli_no_module_level_heavy_imports():
    tree = _modules()["cli.py"]
    banned = ("numpy", "soundfile", "agent_listening.audio", "agent_listening.tasks")
    for mod, line in _top_level_imports(tree):
        if mod in banned or any(mod.startswith(b + ".") for b in banned):
            raise AssertionError(f"cli.py:{line} module-level import of {mod}")

    bad = ast.parse("import numpy\n")
    flagged = [m for m, _ in _top_level_imports(bad) if m == "numpy"]
    assert flagged, "checker must flag numpy"
    lazy = ast.parse("def f():\n    import numpy\n")
    assert not _top_level_imports(lazy), "checker must allow lazy imports inside functions"


def _is_mutable_value(node: ast.expr) -> bool:
    if isinstance(
        node, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    ):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in ("list", "dict", "set", "defaultdict")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr == "defaultdict"
    return False


def test_no_globals_or_module_level_mutables():
    for name, tree in _modules().items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                raise AssertionError(f"{name}:{node.lineno} global/nonlocal statement")
        for node in tree.body:
            if isinstance(node, ast.Assign) and _is_mutable_value(node.value):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if targets != ["__all__"]:
                    raise AssertionError(f"{name}:{node.lineno} module-level mutable container")
            if isinstance(node, ast.AnnAssign) and _is_mutable_value(node.value or ast.Constant(value=None)):
                raise AssertionError(f"{name}:{node.lineno} module-level mutable container")

    bad = ast.parse("X = [1, 2]\ndef f():\n    global X\n")
    assert any(isinstance(n, ast.Global) for n in ast.walk(bad))
    assert any(isinstance(n, ast.Assign) and _is_mutable_value(n.value) for n in bad.body)


_MUTATING_CALLS = (
    "write_bytes",
    "write_text",
    "rename",
    "replace",
    "unlink",
    "touch",
    "mkdir",
    "copy",
    "copyfile",
    "copy2",
    "copytree",
    "move",
    "rmtree",
    "makedirs",
    "removedirs",
    "remove",
    "unlink",
)


def _enclosing_function(tree: ast.Module, line: int) -> str | None:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= line <= (
            node.end_lineno or line
        ):
            best = node.name
    return best


def _mutating_calls(source: str, filename: str) -> list[str]:
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        flagged = None
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                mode = None
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                if isinstance(mode, str) and any(c in mode for c in "wax+"):
                    flagged = "open(write)"
            elif isinstance(func, ast.Attribute) and func.attr in _MUTATING_CALLS:
                flagged = func.attr
            if isinstance(func, ast.Attribute) and func.attr == "fdopen":
                flagged = "os.fdopen"
            if isinstance(func, ast.Attribute) and func.attr in ("write", "writestr"):
                flagged = func.attr
        if flagged:
            func_name = _enclosing_function(tree, node.lineno)
            allowed = (filename == "audio.py" and func_name == "_write_derived") or (
                filename == "cli.py" and func_name == "_write_evidence"
            )
            if filename == "audio.py" and flagged in ("mkdir", "makedirs"):
                allowed = True
            if not allowed:
                hits.append(f"{filename}:{node.lineno} {flagged} inside {func_name}")
    return hits


def test_file_writes_confined():
    for path in sorted(SRC.glob("*.py")):
        hits = _mutating_calls(path.read_text(), path.name)
        assert not hits, f"file writes outside allowed writers: {hits}"

    bad_hits = _mutating_calls("def f():\n    open('x', 'w')\n", "tasks.py")
    assert bad_hits, "checker must flag open('x','w') outside allowed writers"
    good_hits = _mutating_calls("def _write_derived():\n    open('x', 'wb')\n", "audio.py")
    assert not good_hits, "checker must allow writes inside _write_derived"


_BANNED_WORDS = (
    "todo",
    "fixme",
    "xxx",
    "hack",
    "workaround",
    "temporary",
    "for now",
    "no longer",
    "used to",
    "previously",
    "legacy",
    "patched",
    "patch",
    "fixed",
    "fixes",
    "fix",
)
_BANNED_COMMENT = re.compile(r"\b(" + "|".join(_BANNED_WORDS) + r")\b", re.IGNORECASE)


def _comment_hits(text: str) -> list[tuple[int, str]]:
    hits = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT and _BANNED_COMMENT.search(tok.string):
            hits.append((tok.start[0], tok.string))
    return hits


def test_comment_vocabulary():
    for path in sorted(SRC.glob("*.py")):
        hits = _comment_hits(path.read_text())
        assert not hits, f"{path.name} banned comment words: {hits}"
    assert _comment_hits("# todo: rewrite\n"), "checker must flag banned comment words"


def _function_span(tree: ast.Module, name: str) -> tuple[int, int] | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node.lineno, node.end_lineno or node.lineno
    return None


def test_verified_only_in_check():
    word = "ver" + "ified"
    for path in sorted(SRC.glob("*.py")):
        text = path.read_text()
        tree = ast.parse(text)
        span = _function_span(tree, "check") if path.name == "tasks.py" else None
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT) and word in tok.string:
                inside = span is not None and span[0] <= tok.start[0] <= span[1]
                assert inside, f"{path.name}:{tok.start[0]} contains {word!r} outside tasks.check"
    sample = f"x = '{word}'\n"
    assert any(
        word in t.string
        for t in tokenize.generate_tokens(io.StringIO(sample).readline)
        if t.type == tokenize.STRING
    ), "checker must flag the reserved word"


def test_cli_help_loads_no_heavy_modules():
    code = (
        "import sys\n"
        "from agent_listening.cli import main\n"
        "try:\n"
        "    main(['--version'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "heavy = [m for m in ('numpy', 'soundfile') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=SRC.parent,
    )
    assert proc.returncode == 0, proc.stderr


def test_cli_import_loads_no_heavy_modules():
    code = (
        "import sys\n"
        "import agent_listening.cli\n"
        "heavy = [m for m in ('numpy', 'soundfile') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=SRC.parent)
    assert proc.returncode == 0, proc.stderr


def test_listening_prompt_cannot_receive_goals():
    """Goal-blind by API: prompts.build has no way to accept a brief."""
    import inspect

    from agent_listening import prompts

    assert "brief" not in inspect.signature(prompts.build).parameters
    assert "brief" not in inspect.signature(prompts._task_text).parameters
