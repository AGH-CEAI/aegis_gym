import ast
from pathlib import Path

PKG_NAME = "aegis_gym"
REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / PKG_NAME


def module_name(path: Path) -> str:
    parts = path.relative_to(REPO_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def discover_modules() -> dict[str, Path]:
    return {module_name(p): p for p in sorted(PKG_ROOT.rglob("*.py"))}


MODULES = discover_modules()


def is_type_checking_block(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def iter_import_time_nodes(body: list[ast.stmt]):
    """Yield import statements executed at module import time.

    Skips imports inside function bodies and inside
    `if TYPE_CHECKING:` blocks, since those don't run on import.
    """
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        elif isinstance(node, ast.If) and is_type_checking_block(node):
            yield from iter_import_time_nodes(node.orelse)
        else:
            for field in ("body", "orelse", "finalbody", "handlers"):
                children = getattr(node, field, None)
                if not children:
                    continue
                for child in children:
                    if isinstance(child, ast.ExceptHandler):
                        yield from iter_import_time_nodes(child.body)
                    elif isinstance(child, ast.stmt):
                        yield from iter_import_time_nodes([child])


def resolve_relative(current: str, is_package: bool, level: int, name: str | None):
    base = current.split(".")
    if not is_package:
        base = base[:-1]
    if level > 1:
        base = base[: -(level - 1)]
    return ".".join([*base, name] if name else base)


def local_imports(mod: str, path: Path) -> set[str]:
    """Return the in-package modules `mod` imports at import time."""
    tree = ast.parse(path.read_text(), filename=str(path))
    is_package = path.name == "__init__.py"
    deps: set[str] = set()

    for node in iter_import_time_nodes(tree.body):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        else:
            base = (
                resolve_relative(mod, is_package, node.level, node.module)
                if node.level
                else node.module
            )
            # `from pkg import sub` imports the submodule if one exists,
            # otherwise it pulls a name out of `pkg` itself.
            targets = [
                f"{base}.{alias.name}" if f"{base}.{alias.name}" in MODULES else base
                for alias in node.names
            ]
        deps.update(t for t in targets if t in MODULES and t != mod)
    return deps


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Return one representative cycle per strongly connected component."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = low[v] = len(index)
        stack.append(v)
        on_stack.add(v)
        for w in graph[v]:
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                sccs.append(scc)

    for v in sorted(graph):
        if v not in index:
            strongconnect(v)

    cycles = []
    for scc in sccs:
        members = set(scc)
        start = min(scc)
        # BFS back to `start` inside the component for a readable path.
        prev: dict[str, str] = {}
        queue = [start]
        found = None
        while queue and found is None:
            v = queue.pop(0)
            for w in sorted(graph[v] & members):
                if w == start:
                    found = v
                    break
                if w not in prev:
                    prev[w] = v
                    queue.append(w)
        path = [found]
        while path[-1] != start:
            path.append(prev[path[-1]])
        cycles.append([*reversed(path), start])
    return cycles


def test_no_static_circular_imports():
    graph = {mod: local_imports(mod, path) for mod, path in MODULES.items()}
    cycles = find_cycles(graph)

    assert not cycles, "Circular imports found:\n" + "\n".join(
        "  " + " -> ".join(c) for c in cycles
    )
