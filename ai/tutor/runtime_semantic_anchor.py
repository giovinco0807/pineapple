"""Immutable import-completion anchors for Python runtime trust boundaries."""
from __future__ import annotations

import types
from typing import Any, Mapping


ModuleFunctionAnchor = tuple[
    tuple[str, types.FunctionType, types.CodeType, tuple[Any, ...]], ...
]

# The registry is intentionally owned by this independent module.  Producer
# modules register exactly once, at the end of a successful import.  A later
# consumer can therefore distinguish the original import-completion record
# from a freshly rebuilt record made after an in-place function mutation.
_REGISTERED_MODULE_FUNCTION_ANCHORS: dict[
    str, tuple[Mapping[str, Any], ModuleFunctionAnchor]
] = {}


def semantic_state(
    value: Any,
    active: set[int] | None = None,
) -> tuple[Any, ...]:
    """Describe code, defaults, closures, and wrappers without a mutable hash."""

    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if value is Ellipsis:
        return ("ellipsis",)
    if value is NotImplemented:
        return ("not_implemented",)

    seen = active if active is not None else set()
    identity = id(value)
    if identity in seen:
        value_type = type(value)
        return (
            "recursive_reference",
            value_type.__module__,
            value_type.__qualname__,
        )
    seen.add(identity)
    try:
        if isinstance(value, types.CodeType):
            return (
                "code",
                value.co_argcount,
                value.co_posonlyargcount,
                value.co_kwonlyargcount,
                value.co_nlocals,
                value.co_stacksize,
                value.co_flags,
                value.co_code,
                getattr(value, "co_exceptiontable", b""),
                tuple(semantic_state(item, seen) for item in value.co_consts),
                tuple(value.co_names),
                tuple(value.co_varnames),
                tuple(value.co_freevars),
                tuple(value.co_cellvars),
            )
        if isinstance(value, types.FunctionType):
            closure: list[tuple[Any, ...]] = []
            for cell in value.__closure__ or ():
                try:
                    contents = cell.cell_contents
                except ValueError:
                    closure.append(("empty_cell",))
                else:
                    closure.append(semantic_state(contents, seen))
            wrapped = getattr(value, "__wrapped__", None)
            return (
                "python_function",
                value.__module__,
                value.__qualname__,
                value.__name__,
                semantic_state(value.__code__, seen),
                semantic_state(value.__defaults__, seen),
                semantic_state(value.__kwdefaults__, seen),
                tuple(closure),
                (
                    ("none",)
                    if wrapped is None
                    else semantic_state(wrapped, seen)
                ),
            )
        if isinstance(value, types.MethodType):
            owner = value.__self__
            owner_type = owner if isinstance(owner, type) else type(owner)
            return (
                "bound_method",
                semantic_state(value.__func__, seen),
                owner_type.__module__,
                owner_type.__qualname__,
            )
        if isinstance(value, type):
            return ("class", value.__module__, value.__qualname__)
        if isinstance(value, tuple):
            return (
                "tuple",
                tuple(semantic_state(item, seen) for item in value),
            )
        if isinstance(value, list):
            return (
                "list",
                tuple(semantic_state(item, seen) for item in value),
            )
        if isinstance(value, (set, frozenset)):
            items = [semantic_state(item, seen) for item in value]
            items.sort(key=repr)
            return (type(value).__name__, tuple(items))
        if isinstance(value, Mapping):
            items = [
                (semantic_state(key, seen), semantic_state(item, seen))
                for key, item in value.items()
            ]
            items.sort(key=lambda pair: repr(pair[0]))
            return ("mapping", tuple(items))
        if isinstance(value, types.ModuleType):
            return ("module", value.__name__)
        if callable(value):
            owner_class = getattr(value, "__objclass__", None)
            return (
                "native_callable",
                type(value).__module__,
                type(value).__qualname__,
                getattr(value, "__module__", None),
                getattr(value, "__qualname__", None),
                getattr(value, "__name__", None),
                getattr(value, "__text_signature__", None),
                (
                    None
                    if owner_class is None
                    else (owner_class.__module__, owner_class.__qualname__)
                ),
            )
        raise TypeError(
            "unsupported runtime semantic value: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )
    finally:
        seen.remove(identity)


def build_module_function_anchor(
    namespace: Mapping[str, Any],
) -> ModuleFunctionAnchor:
    """Capture every Python function bound when a module finishes importing."""

    return tuple(
        (name, target, target.__code__, semantic_state(target))
        for name, target in sorted(namespace.items())
        if isinstance(target, types.FunctionType)
    )


def register_module_function_anchor(
    module_name: str,
    namespace: Mapping[str, Any],
) -> ModuleFunctionAnchor:
    """Register one immutable import-completion record for a module.

    Registration is deliberately one-shot.  Re-import-time replacement under
    the same module name, or reuse of one namespace under another name, fails
    closed instead of silently establishing a new baseline.
    """

    if not isinstance(module_name, str) or not module_name:
        raise RuntimeError("module semantic registry name must be nonempty")
    if not isinstance(namespace, Mapping):
        raise RuntimeError("module semantic registry namespace must be a mapping")
    if namespace.get("__name__") != module_name:
        raise RuntimeError("module semantic registry namespace/name mismatch")
    if module_name in _REGISTERED_MODULE_FUNCTION_ANCHORS:
        raise RuntimeError(
            f"module semantic registry duplicate registration: {module_name}"
        )
    if any(
        registered_namespace is namespace
        for registered_namespace, _anchor in (
            _REGISTERED_MODULE_FUNCTION_ANCHORS.values()
        )
    ):
        raise RuntimeError("module semantic registry namespace rebind")
    anchor = build_module_function_anchor(namespace)
    if not anchor:
        raise RuntimeError("module semantic registry anchor must be nonempty")
    _REGISTERED_MODULE_FUNCTION_ANCHORS[module_name] = (namespace, anchor)
    return anchor


def registered_module_function_anchor(
    module_name: str,
    namespace: Mapping[str, Any],
) -> ModuleFunctionAnchor:
    """Return the original record only for its exact registered namespace."""

    if not isinstance(module_name, str) or not module_name:
        raise RuntimeError("module semantic registry name must be nonempty")
    registered = _REGISTERED_MODULE_FUNCTION_ANCHORS.get(module_name)
    if registered is None:
        raise RuntimeError(
            f"module semantic registry record missing: {module_name}"
        )
    registered_namespace, anchor = registered
    if registered_namespace is not namespace:
        raise RuntimeError(
            f"module semantic registry namespace drift: {module_name}"
        )
    return anchor


def verify_module_function_anchor(
    namespace: Mapping[str, Any],
    anchor: ModuleFunctionAnchor,
) -> None:
    """Fail if an anchored alias or any mutable function semantic has drifted."""

    if not isinstance(anchor, tuple) or not anchor:
        raise RuntimeError("module semantic anchor must be a nonempty tuple")
    names: set[str] = set()
    for entry in anchor:
        if not isinstance(entry, tuple) or len(entry) != 4:
            raise RuntimeError("module semantic anchor entry malformed")
        name, canonical, original_code, expected_state = entry
        if not isinstance(name, str) or not name or name in names:
            raise RuntimeError("module semantic anchor name malformed or duplicated")
        names.add(name)
        current = namespace.get(name)
        if current is not canonical:
            raise RuntimeError(f"module semantic anchor alias drift: {name}")
        if current.__code__ is not original_code:
            raise RuntimeError(f"module semantic anchor code drift: {name}")
        if semantic_state(current) != expected_state:
            raise RuntimeError(f"module semantic anchor state drift: {name}")


__all__ = [
    "ModuleFunctionAnchor",
    "build_module_function_anchor",
    "register_module_function_anchor",
    "registered_module_function_anchor",
    "semantic_state",
    "verify_module_function_anchor",
]
