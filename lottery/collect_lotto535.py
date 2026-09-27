"""Collect Lotto 5/35 results (5 main numbers + the special number) with the jackpot and every tier's winners and prize
from vietlott.vn. On a jackpot share-out draw the prize column already includes each tier's share.

Usage: python3 lottery/collect_lotto535.py            # fetch any draws not yet saved
       python3 lottery/collect_lotto535.py --workers 4
"""

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import collect_prizes as c

BASE = "https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/535"
OUT = Path(__file__).resolve().parent.parent / "data" / "lotto535.jsonl"
TIERS = {"Giải Độc Đắc": "jackpot", "Giải Nhất": "first", "Giải Nhì": "second", "Giải Ba": "third",
         "Giải Tư": "fourth", "Giải Năm": "fifth", "Giải Khuyến Khích": "consolation"}


def parse(html, want_id=None):
    m = re.search(r"Kỳ quay thưởng\s*<b>#(\d+)</b>\s*ngày\s*<b>(\d\d)/(\d\d)/(\d{4})</b>", html)
    if not m or (want_id is not None and int(m.group(1)) != want_id):
        raise ValueError("draw header missing or id mismatch")
    block = html[m.end(): m.end() + 2000]
    balls = [int(x) for x in re.findall(r'<span class="bong_tron[^"]*">(\d+)</span>', block)]
    if len(balls) != 6:
        raise ValueError(f"expected 6 balls, got {balls}")
    row = {"id": int(m.group(1)), "date": f"{m.group(4)}-{m.group(3)}-{m.group(2)}",
           "result": sorted(balls[:5]), "special": balls[5]}
    for tr in re.findall(r"<tr>(.*?)</tr>", html, re.S):
        cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", x)).strip()
                 for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cells) == 4 and cells[0] in TIERS:
            row[f"{TIERS[cells[0]]}_winners"] = c._vnd(cells[2])
            row[f"{TIERS[cells[0]]}_prize"] = c._vnd(cells[3])
    missing = [k for k in TIERS.values() if f"{k}_winners" not in row]
    if missing:
        raise ValueError(f"prize rows missing: {missing}")
    return row


def latest_id():
    return parse(c._get(BASE))["id"]


def fetch(draw_id):
    row = parse(c._get(f"{BASE}?id={draw_id:05d}&nocatche=1"), draw_id)
    time.sleep(0.5)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rows = c.load(OUT)
    last = latest_id()
    todo = [i for i in range(1, last + 1) if i not in rows]
    print(f"have {len(rows)} draws, latest is #{last}, fetching {len(todo)}", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, i): i for i in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
                rows[row["id"]] = row
            except Exception as e:
                failed.append((futures[fut], str(e)))
            if n % 100 == 0:
                c.save(rows, OUT)
                print(f"  {n}/{len(todo)}", flush=True)
    c.save(rows, OUT)
    print(f"saved {len(rows)} draws to {OUT}" + (f"; failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
