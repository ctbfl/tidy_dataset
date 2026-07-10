"""Run one scenario-brainstorm experiment end to end.

Builds the prompt (build_scenario_prompt.build_prompt), sends it to the VLM via
organize_it.modules.vlm.codex, and saves the prompt + raw response (+ parsed
narratives + meta) under result/<exp_name>/.

用法:
    python run_brainstorm.py --direction "客厅茶几 coffee_table" --exp coffee_table_nodedup_01
    python run_brainstorm.py --direction "书桌 study_desk" --dedup old.txt --n 6 --exp study_desk_01
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORGANIZE_IT_SRC = Path("/home/hjs/Projects/table_arrangement/organize_it_v2/src")
if str(ORGANIZE_IT_SRC) not in sys.path:
    sys.path.insert(0, str(ORGANIZE_IT_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_scenario_prompt import build_prompt  # noqa: E402
from organize_it.modules import vlm  # noqa: E402

RESULT_DIR = HERE / "result"


def extract_narratives(text: str) -> list[str] | None:
    """从模型输出里抽出那段 JSON 字符串数组(优先 ```json 代码块,回退到最后一个 [...] )。"""
    blocks = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    candidates = list(blocks)
    if not candidates:
        candidates = re.findall(r"(\[[^\[\]]*\])", text, flags=re.DOTALL)
    for chunk in reversed(candidates):
        try:
            arr = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return [s.strip() for s in arr]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True)
    ap.add_argument("--dedup", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--exp", required=True, help="实验名(结果保存到 result/<exp>/)")
    ap.add_argument("--model", default=vlm.OPENAI_MODEL)
    ap.add_argument("--effort", default=None, help="reasoning effort (low/medium/high)")
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()

    out_dir = RESULT_DIR / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(args.direction, dedup=args.dedup, n=args.n)
    (out_dir / "prompt.md").write_text(prompt)
    print(f"[prompt] {len(prompt)} chars -> {out_dir/'prompt.md'}")

    print(f"[vlm] calling {args.model} …")
    t0 = time.time()
    response = vlm.codex(
        prompt,
        model=args.model,
        reasoning_effort=args.effort,
        timeout=args.timeout,
    )
    dt = time.time() - t0
    (out_dir / "response.md").write_text(response)
    print(f"[vlm] {len(response)} chars in {dt:.1f}s -> {out_dir/'response.md'}")

    narratives = extract_narratives(response)
    if narratives is not None:
        (out_dir / "narratives.json").write_text(
            json.dumps(narratives, ensure_ascii=False, indent=2)
        )
        print(f"[parse] extracted {len(narratives)} narratives -> {out_dir/'narratives.json'}")
    else:
        print("[parse] WARNING: could not extract a JSON string array from the response")

    meta = {
        "exp": args.exp,
        "direction": args.direction,
        "dedup": args.dedup,
        "n": args.n,
        "model": args.model,
        "reasoning_effort": args.effort,
        "base_url": vlm.OPENAI_BASE_URL,
        "elapsed_sec": round(dt, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "narrative_count": len(narratives) if narratives else 0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[meta] -> {out_dir/'meta.json'}")

    if narratives:
        print("\n=== narratives ===")
        for i, s in enumerate(narratives, 1):
            print(f"  {i}. {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
