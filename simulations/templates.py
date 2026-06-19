# Scenario templates: a template declares the *roles* on a desk (how many of
# what), each role drawing from a candidate pool of concrete assets. Generation
# samples one manifest per scene from this; the editor only round-trips the
# resulting manifest, so it never needs this module.

from __future__ import annotations

import json
from pathlib import Path

from objects import AssetLibrary

REPO = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = REPO / "templates"


def load_template(template: str) -> dict:
    """Accept a template id (templates/<id>.json) or a direct path."""
    path = Path(template)
    if not path.exists():
        path = TEMPLATES_DIR / f"{template}.json"
    return json.loads(path.read_text())


def resolve_candidates(library: AssetLibrary, role: dict) -> list[str]:
    """Candidate asset ids for a role = (assets matching any of role['tags'])
    ∪ role['asset_ids'], de-duplicated, tags first then explicit ids, finally
    dropping anything in role['exclude_ids'] or whose id contains an
    role['exclude_kw'] substring (used to prune tag false-positives, e.g. a
    'cup' role excluding 'measuring'/'ramekin'), and any asset currently
    disabled (semantics.enabled == false, checked live against disk)."""
    ids: list[str] = []
    seen: set[str] = set()

    def add(aid: str) -> None:
        if aid not in seen:
            seen.add(aid)
            ids.append(aid)

    for tag in role.get("tags", []):
        for asset in library.by_tag(tag):
            add(asset.id)
    for aid in role.get("asset_ids", []):
        add(aid)

    excl_ids = set(role.get("exclude_ids", []))
    excl_kw = [k.lower() for k in role.get("exclude_kw", [])]
    return [i for i in ids
            if i not in excl_ids
            and not any(k in i.lower() for k in excl_kw)
            and library.is_enabled(i)]


def _draw_count(count, rng) -> int:
    if isinstance(count, (list, tuple)):
        lo, hi = count
        return rng.randint(int(lo), int(hi))
    return int(count)


def _weighted_choice(options: list[dict], rng) -> dict:
    """Pick one option, probability proportional to its 'weight' (default 1)."""
    weights = [float(o.get("weight", 1)) for o in options]
    return rng.choices(options, weights=weights, k=1)[0]


def sample_manifest(template: dict, library: AssetLibrary, rng) -> list[dict]:
    """Sample one scene's manifest.

    Two sources of roles, both optional:
      - template['groups']: each group is a weighted set of mutually-exclusive
        options; exactly one option is drawn and its roles emitted. Use this for
        correlated choices (e.g. a workstation = laptop-only | laptop+mouse |
        display+keyboard+mouse | laptop+keyboard+mouse).
      - template['roles']: independent roles, each emitted on its own.

    For every emitted role we draw a count, then pick that many *distinct* assets
    from its candidate pool. Slots are role-1, role-2, ... Returns a list of
    {slot, role, asset_id}; roles with an empty pool or a drawn count of 0 add nothing.
    """
    manifest: list[dict] = []
    counts: dict[str, int] = {}  # role name -> running slot index

    def emit(role: dict) -> None:
        pool = resolve_candidates(library, role)
        if not pool:
            return
        n = min(_draw_count(role["count"], rng), len(pool))
        for asset_id in rng.sample(pool, n):
            counts[role["role"]] = counts.get(role["role"], 0) + 1
            manifest.append({"slot": f"{role['role']}-{counts[role['role']]}",
                             "role": role["role"], "asset_id": asset_id})

    for group in template.get("groups", []):
        for role in _weighted_choice(group["options"], rng).get("roles", []):
            emit(role)
    for role in template.get("roles", []):
        emit(role)
    return manifest
