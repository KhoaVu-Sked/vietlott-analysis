"""Collect Mega 6/45 results plus per-tier winner counts from vietlott.vn.

Usage: python3 lottery/collect_prizes.py            # fetch any draws not yet saved
       python3 lottery/collect_prizes.py --workers 4
"""

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
BASE = "https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/645"
OUT = Path(__file__).resolve().parent.parent / "data" / "mega645.jsonl"
TIERS = {"Jackpot": "jackpot", "Giải Nhất": "first", "Giải Nhì": "second", "Giải Ba": "third"}


def _vnd(s):
    return int(s.replace(".", "").strip())


def _get(url, tries=4):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def parse(html, want_id=None):
    m = re.search(r"Kỳ quay thưởng\s*<b>#(\d+)</b>\s*ngày\s*<b>(\d\d)/(\d\d)/(\d{4})</b>", html)
    if not m or (want_id is not None and int(m.group(1)) != want_id):
        raise ValueError("draw header missing or id mismatch")
    block = html[m.end(): m.end() + 1500]
    nums = [int(x) for x in re.findall(r'<span class="bong_tron[^"]*">(\d+)</span>', block)]
    if len(nums) != 6:
        raise ValueError(f"expected 6 numbers, got {nums}")
    jp = re.search(r'<div class="so_tien">\s*<h3>([\d.]+)</h3>', html)
    row = {"id": int(m.group(1)), "date": f"{m.group(4)}-{m.group(3)}-{m.group(2)}",
           "result": sorted(nums), "jackpot_value": _vnd(jp.group(1)) if jp else None}
    for tr in re.findall(r"<tr>(.*?)</tr>", html, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cells) == 4 and cells[0] in TIERS:
            row[f"{TIERS[cells[0]]}_winners"] = _vnd(cells[2])
            row[f"{TIERS[cells[0]]}_prize"] = _vnd(cells[3])
    missing = [k for k in TIERS.values() if f"{k}_winners" not in row]
    if missing:
        raise ValueError(f"prize rows missing: {missing}")
    return row


def latest_id():
    return parse(_get(BASE))["id"]


def fetch(draw_id):
    row = parse(_get(f"{BASE}?id={draw_id:05d}&nocatche=1"), draw_id)
    time.sleep(0.5)
    return row


def load(path=OUT):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def save(rows, path=OUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(rows[k], ensure_ascii=False) + "\n" for k in sorted(rows)))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rows = load()
    last = latest_id()
    todo = [i for i in range(1, last + 1) if i not in rows]
    print(f"have {len(rows)} draws, latest is #{last}, fetching {len(todo)}")
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, i): i for i in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
                rows[row["id"]] = row
            except Exception as e:
                failed.append((futures[fut], str(e)))
            if n % 50 == 0:
                save(rows)
                print(f"  {n}/{len(todo)}")
    save(rows)
    print(f"saved {len(rows)} draws to {OUT}" + (f"; failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
