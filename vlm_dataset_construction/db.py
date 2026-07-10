"""Single source of truth (SQLite) for VLM-constructed scenario templates.

三个层级 = 三层表(count_template 再拆出 items):

  scenario            大的场景方向(纯文本 direction)
    └─ variation      叙事(纯文本,字段名就叫 `variation`)
         └─ count_template     一份"物体数量清单"(无 layout);一条 variation 可挂多份
              └─ count_template_item   category_id : count

去重都从这里查:`ScenarioDB.existing_narratives()` 直接喂给 build_scenario_prompt 的 dedup。

用法:
    from db import ScenarioDB
    db = ScenarioDB()
    sid = db.get_or_create_scenario("coffee_table", direction="客厅茶几 coffee_table", name="客厅茶几")
    vid = db.add_variation("coffee_table", "下班后年轻人瘫沙发看综艺,边吃零食边喝饮料。", source={"exp": "..."})
    tid = db.add_count_template(vid, [{"category": "can_drink", "count": 2}, {"category": "snack_box", "count": 1}])
    tid2 = db.add_count_template(vid, [{"category": "chip_can", "count": 1}])   # 同一 variation 的第二份清单
    dedup = db.existing_narratives("coffee_table")

CLI:
    python db.py init
    python db.py import-exp --exp coffee_table_nodedup_01 --scenario coffee_table --name 客厅茶几 --direction "客厅茶几 coffee_table"
    python db.py narratives --scenario coffee_table [--json]
    python db.py show [--scenario coffee_table]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_scenario_prompt import load_categories  # noqa: E402  (类目校验的唯一真源)

DB_PATH = HERE / "store.db"
RESULT_DIR = HERE / "result"

SCHEMA = """
CREATE TABLE IF NOT EXISTS scenario (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    slug        TEXT UNIQUE NOT NULL,          -- 短句柄,便于引用(coffee_table)
    direction   TEXT NOT NULL,                 -- 大的场景方向,纯文本
    name        TEXT DEFAULT '',
    description TEXT DEFAULT '',
    created     TEXT
);
CREATE TABLE IF NOT EXISTS variation (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id INTEGER NOT NULL REFERENCES scenario(id) ON DELETE CASCADE,
    variation   TEXT NOT NULL,                 -- 叙事,纯文本
    slug        TEXT DEFAULT '',               -- 简写英文名,用作数据集 variation 文件夹名
    norm        TEXT NOT NULL,                 -- 归一化(去重用)
    status      TEXT DEFAULT 'draft',          -- draft | accepted | rejected
    source      TEXT DEFAULT '',               -- json 字符串(哪次实验/模型)
    created     TEXT,
    UNIQUE(scenario_id, norm)
);
CREATE TABLE IF NOT EXISTS count_template (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    variation_id INTEGER NOT NULL REFERENCES variation(id) ON DELETE CASCADE,
    instructions TEXT DEFAULT '',            -- 模拟用户对整理机器人说的需求语句(不含摆放指令)
    note         TEXT DEFAULT '',
    status       TEXT DEFAULT 'draft',
    created      TEXT
);
CREATE TABLE IF NOT EXISTS count_template_item (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    template_id INTEGER NOT NULL REFERENCES count_template(id) ON DELETE CASCADE,
    category_id TEXT NOT NULL,
    count       INTEGER NOT NULL CHECK(count >= 1)
);
CREATE INDEX IF NOT EXISTS idx_variation_scenario ON variation(scenario_id);
CREATE INDEX IF NOT EXISTS idx_template_variation ON count_template(variation_id);
CREATE INDEX IF NOT EXISTS idx_item_template ON count_template_item(template_id);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize(text: str) -> str:
    """去重比较:去空白 + 全角/半角标点归一化 + 小写。"""
    t = "".join(text.split())
    table = str.maketrans({"，": ",", "。": ".", "、": ",", "；": ";", "：": ":",
                           "（": "(", "）": ")", "！": "!", "？": "?"})
    return t.translate(table).lower()


class ScenarioDB:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, timeout=30)  # 并行写库时等待锁而非报错
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """老库补列(idempotent)。"""
        ct_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(count_template)")}
        if "instructions" not in ct_cols:
            self.conn.execute("ALTER TABLE count_template ADD COLUMN instructions TEXT DEFAULT ''")
        v_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(variation)")}
        if "slug" not in v_cols:
            self.conn.execute("ALTER TABLE variation ADD COLUMN slug TEXT DEFAULT ''")

    def set_variation_slug(self, variation_id: int, slug: str) -> None:
        self.conn.execute("UPDATE variation SET slug=? WHERE id=?", (slug.strip(), variation_id))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------------- scenario ----------------
    def get_or_create_scenario(self, slug: str, *, direction="", name="", description="") -> int:
        row = self.conn.execute("SELECT * FROM scenario WHERE slug=?", (slug,)).fetchone()
        if row:
            # 回填后来才提供的元数据
            updates = {k: v for k, v in (("direction", direction), ("name", name),
                                         ("description", description)) if v and not row[k]}
            if updates:
                sets = ", ".join(f"{k}=?" for k in updates)
                self.conn.execute(f"UPDATE scenario SET {sets} WHERE id=?",
                                  (*updates.values(), row["id"]))
                self.conn.commit()
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO scenario(slug, direction, name, description, created) VALUES(?,?,?,?,?)",
            (slug, direction or slug, name, description, _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def _scenario_id(self, scenario) -> int:
        if isinstance(scenario, int):
            return scenario
        row = self.conn.execute("SELECT id FROM scenario WHERE slug=?", (scenario,)).fetchone()
        if not row:
            raise ValueError(f"scenario 不存在: {scenario!r}")
        return row["id"]

    def get_scenario(self, scenario) -> dict | None:
        col = "id" if isinstance(scenario, int) else "slug"
        row = self.conn.execute(f"SELECT * FROM scenario WHERE {col}=?", (scenario,)).fetchone()
        return dict(row) if row else None

    def list_scenarios(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM scenario ORDER BY slug")]

    # ---------------- variation ----------------
    def add_variation(self, scenario, narrative: str, *, status="draft", source=None,
                      dedup=True) -> int | None:
        """新增叙事;dedup=True 且已有等价叙事时跳过并返回 None。"""
        narrative = narrative.strip()
        if not narrative:
            raise ValueError("narrative 不能为空")
        sid = self._scenario_id(scenario)
        norm = _normalize(narrative)
        if dedup:
            hit = self.conn.execute(
                "SELECT id FROM variation WHERE scenario_id=? AND norm=?", (sid, norm)
            ).fetchone()
            if hit:
                return None
        src = json.dumps(source or {}, ensure_ascii=False)
        try:
            cur = self.conn.execute(
                "INSERT INTO variation(scenario_id, variation, norm, status, source, created)"
                " VALUES(?,?,?,?,?,?)",
                (sid, narrative, norm, status, src, _now()),
            )
        except sqlite3.IntegrityError:
            return None  # UNIQUE(scenario_id, norm)
        self.conn.commit()
        return cur.lastrowid

    def list_variations(self, scenario=None) -> list[dict]:
        if scenario is None:
            rows = self.conn.execute("SELECT * FROM variation ORDER BY scenario_id, id")
        else:
            rows = self.conn.execute("SELECT * FROM variation WHERE scenario_id=? ORDER BY id",
                                     (self._scenario_id(scenario),))
        return [dict(r) for r in rows]

    def get_variation(self, variation_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM variation WHERE id=?", (variation_id,)).fetchone()
        return dict(row) if row else None

    def set_variation_status(self, variation_id: int, status: str) -> None:
        self.conn.execute("UPDATE variation SET status=? WHERE id=?", (status, variation_id))
        self.conn.commit()

    # ---------------- count_template ----------------
    def add_count_template(self, variation_id: int, objects, *, instructions="", note="",
                           status="draft", validate=True) -> int:
        """给一条 variation 追加一份物体数量清单 + 用户需求语句。

        objects: [{category,count}] 或 [(category,count)]。
        instructions: 模拟用户对整理机器人说的需求语句(不含具体摆放指令)。
        """
        if not self.get_variation(variation_id):
            raise ValueError(f"variation 不存在: {variation_id}")
        cats = load_categories() if validate else None
        items = []
        for o in objects:
            if isinstance(o, dict):
                cat, cnt = str(o["category"]).strip(), int(o["count"])
            else:
                cat, cnt = str(o[0]).strip(), int(o[1])
            if cnt < 1:
                raise ValueError(f"count 必须 >=1: {o}")
            if validate and cat not in cats:
                raise ValueError(f"未知类目(不在 available_assets.json): {cat}")
            items.append((cat, cnt))
        cur = self.conn.execute(
            "INSERT INTO count_template(variation_id, instructions, note, status, created)"
            " VALUES(?,?,?,?,?)",
            (variation_id, instructions, note, status, _now()),
        )
        tid = cur.lastrowid
        self.conn.executemany(
            "INSERT INTO count_template_item(template_id, category_id, count) VALUES(?,?,?)",
            [(tid, c, n) for c, n in items],
        )
        self.conn.commit()
        return tid

    def list_count_templates(self, variation_id: int) -> list[dict]:
        out = []
        for t in self.conn.execute("SELECT * FROM count_template WHERE variation_id=? ORDER BY id",
                                   (variation_id,)):
            items = self.conn.execute(
                "SELECT category_id, count FROM count_template_item WHERE template_id=? ORDER BY id",
                (t["id"],),
            ).fetchall()
            d = dict(t)
            d["objects"] = [{"category": i["category_id"], "count": i["count"]} for i in items]
            out.append(d)
        return out

    # ---------------- dedup / import ----------------
    def existing_narratives(self, scenario=None, statuses: tuple[str, ...] | None = None) -> list[str]:
        rows = self.list_variations(scenario)
        return [r["variation"] for r in rows if statuses is None or r["status"] in statuses]

    def import_experiment(self, exp_name: str, scenario_slug: str, *,
                          name="", direction="", dedup=True) -> list[int]:
        narr_file = RESULT_DIR / exp_name / "narratives.json"
        if not narr_file.is_file():
            raise FileNotFoundError(narr_file)
        meta_file = RESULT_DIR / exp_name / "meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
        self.get_or_create_scenario(scenario_slug, direction=direction or meta.get("direction", ""),
                                    name=name)
        added = []
        for narrative in json.loads(narr_file.read_text()):
            vid = self.add_variation(scenario_slug, narrative,
                                     source={"exp": exp_name, "model": meta.get("model", "")},
                                     dedup=dedup)
            if vid is not None:
                added.append(vid)
        return added


# ----------------------------- CLI -----------------------------
def _cli() -> int:
    ap = argparse.ArgumentParser(description="scenario/variation/count_template 单一真源 (SQLite)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="创建/初始化数据库")

    p_imp = sub.add_parser("import-exp", help="把某次 run_brainstorm 的 narratives 导入")
    p_imp.add_argument("--exp", required=True)
    p_imp.add_argument("--scenario", required=True)
    p_imp.add_argument("--name", default="")
    p_imp.add_argument("--direction", default="")
    p_imp.add_argument("--no-dedup", action="store_true")

    p_nar = sub.add_parser("narratives", help="打印某 scenario 的全部叙事(去重用)")
    p_nar.add_argument("--scenario", default=None)
    p_nar.add_argument("--json", action="store_true")

    p_show = sub.add_parser("show", help="打印三层树")
    p_show.add_argument("--scenario", default=None)

    args = ap.parse_args()
    db = ScenarioDB()

    if args.cmd == "init":
        print(f"initialized {db.path}")
        return 0

    if args.cmd == "import-exp":
        added = db.import_experiment(args.exp, args.scenario, name=args.name,
                                     direction=args.direction, dedup=not args.no_dedup)
        print(f"imported {len(added)} new variations into '{args.scenario}'")
        for vid in added:
            print(f"  v{vid}  {db.get_variation(vid)['variation']}")
        return 0

    if args.cmd == "narratives":
        items = db.existing_narratives(args.scenario)
        print(json.dumps(items, ensure_ascii=False, indent=2) if args.json
              else "\n".join(f"- {x}" for x in items))
        return 0

    if args.cmd == "show":
        scns = [db.get_scenario(args.scenario)] if args.scenario else db.list_scenarios()
        for scn in scns:
            if not scn:
                continue
            print(f"■ [{scn['id']}] {scn['slug']}  ({scn['name']})  — {scn['direction']}")
            for v in db.list_variations(scn["slug"]):
                print(f"    └─ v{v['id']} [{v['status']}]  {v['variation']}")
                for t in db.list_count_templates(v["id"]):
                    objs = ", ".join(f"{o['category']}×{o['count']}" for o in t["objects"])
                    print(f"          └─ t{t['id']} [{t['status']}]")
                    if t.get("instructions"):
                        print(f"               需求: {t['instructions']}")
                    print(f"               物体: {objs}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
