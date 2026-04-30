"""Helpers for resolving explicit DA-scope selections."""

from __future__ import annotations

from pathlib import Path


NAMED_DA_SCOPES: dict[str, tuple[str, ...]] = {}


def resolve_da_scope_codes(
    *,
    da_codes: list[str] | tuple[str, ...] | None,
    da_scope_name: str | None,
    da_codes_file: str | Path | None,
) -> list[str] | None:
    """Resolve explicit DA codes from direct input, file input, or named scope."""
    if da_codes is not None:
        cleaned = [str(code).strip() for code in da_codes if str(code).strip()]
        return cleaned
    if da_codes_file is not None:
        path = Path(da_codes_file)
        values = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return values
    if da_scope_name is not None:
        if da_scope_name not in NAMED_DA_SCOPES:
            raise KeyError(f"Unknown DA scope name: {da_scope_name}")
        return list(NAMED_DA_SCOPES[da_scope_name])
    return None
