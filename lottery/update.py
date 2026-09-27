"""Bring the saved draws up to date with vietlott.vn before anything uses them."""

import collect_lotto535 as cl
import collect_power655 as cp
import collect_prizes as c

SOURCES = {"645": (c.OUT, c.fetch, c.latest_id), "655": (cp.OUT, cp.fetch, cp.latest_id),
           "535": (cl.OUT, cl.fetch, cl.latest_id)}


def refresh(game):
    out, fetch, latest = SOURCES[game]
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
