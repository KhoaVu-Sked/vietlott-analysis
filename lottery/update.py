"""Bring the saved draws up to date with vietlott.vn before anything uses them."""

import collect_power655 as cp
import collect_prizes as c


def refresh(game):
    out, fetch, latest = (c.OUT, c.fetch, c.latest_id) if game == "645" else (cp.OUT, cp.fetch, cp.latest_id)
    try:
        rows = c.load(out)
        missing = list(range(max(rows, default=0) + 1, latest() + 1))
        for i in missing:
            rows[i] = fetch(i)
        if missing:
            c.save(rows, out)
        return f"up to date with vietlott.vn ({len(missing)} new draw{'s' if len(missing) != 1 else ''} added)"
    except Exception as e:
        return f"could not reach vietlott.vn ({type(e).__name__}), using the saved data"
