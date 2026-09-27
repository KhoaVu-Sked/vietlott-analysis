"""A small terminal UI in the style of Claude Code: rounded boxes, arrow-key menus, an input box and a spinner."""

import os
import re
import shutil
import sys
import threading
import time

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
FRAMES = "·✢✳✶✻✽✻✶✳✢"
KEYS = {b"\x1b[A": "up", b"\x1b[B": "down", b"\x1b[C": "right", b"\x1b[D": "left",
        b"\x1bOA": "up", b"\x1bOB": "down", b"\x1bOC": "right", b"\x1bOD": "left"}


def visible_len(text):
    return len(ANSI.sub("", text))


def pad(text, width):
    return text + " " * max(0, width - visible_len(text))


class Style:
    def __init__(self):
        mode = None
        if sys.stdout.isatty() and not os.environ.get("NO_COLOR"):
            mode = "true" if os.environ.get("COLORTERM") in ("truecolor", "24bit") else "256"

        def fg(rgb, code):
            if mode == "true":
                return "\x1b[38;2;{};{};{}m".format(*rgb)
            return f"\x1b[38;5;{code}m" if mode == "256" else ""

        self.on = mode is not None
        self.accent = fg((215, 119, 87), 173)
        self.green = fg((78, 186, 101), 71)
        self.yellow = fg((230, 180, 80), 179)
        self.red = fg((230, 90, 90), 167)
        self.grey = fg((136, 136, 136), 245)
        self.dim = "\x1b[2m" if self.on else ""
        self.bold = "\x1b[1m" if self.on else ""
        self.inverse = "\x1b[7m" if self.on else ""
        self.reset = "\x1b[0m" if self.on else ""

    def __call__(self, text, *codes):
        return "".join(codes) + text + self.reset if self.on and any(codes) else text


class Terminal:
    def __init__(self):
        self.style = Style()
        self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self.drawn = 0
        self.fd = sys.stdin.fileno() if self.interactive else None
        self.saved = None

    @property
    def width(self):
        return max(40, min(shutil.get_terminal_size((80, 24)).columns - 1, 88))

    def __enter__(self):
        if self.interactive:
            import termios
            import tty
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.write("\x1b[?25l")
        return self

    def __exit__(self, *exc):
        if self.interactive and self.saved is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.write("\x1b[?25h")
        return False

    def write(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()

    def clear(self):
        if self.interactive:
            self.write("\x1b[H\x1b[2J\x1b[3J")
        self.drawn = 0

    def print(self, *lines):
        self.write("".join(line + "\n" for line in lines))

    def live(self, lines):
        if self.drawn:
            self.write(f"\r\x1b[{self.drawn}A\x1b[J")
        self.print(*lines)
        self.drawn = len(lines)

    def settle(self, *lines):
        self.live(list(lines))
        self.drawn = 0
        return lines[0] if lines else None

    def keys(self):
        buf = os.read(self.fd, 64)
        out, i = [], 0
        while i < len(buf):
            for seq, name in KEYS.items():
                if buf.startswith(seq, i):
                    out.append(name)
                    i += len(seq)
                    break
            else:
                c = buf[i:i + 1]
                i += 1
                if c == b"\x1b":
                    if buf[i:i + 1] in (b"[", b"O"):
                        i += 1
                        while i < len(buf) and not (64 <= buf[i] <= 126):
                            i += 1
                        i += 1
                    else:
                        out.append("esc")
                elif c in (b"\r", b"\n"):
                    out.append("enter")
                elif c in (b"\x7f", b"\x08"):
                    out.append("backspace")
                elif c in (b"\x03", b"\x04"):
                    raise KeyboardInterrupt
                else:
                    out.append(c.decode("utf-8", "ignore"))
        return out

    def box(self, rows, title="", colour=""):
        s, w = self.style, self.width
        inner = w - 4
        head = f"─ {title} " if title else ""
        top = s("╭" + head, colour) + s("─" * (w - 2 - visible_len(head)) + "╮", colour)
        body = [s("│", colour) + " " + pad(row, inner) + " " + s("│", colour) for row in rows]
        return [top, *body, s("╰" + "─" * (w - 2) + "╯", colour)]

    def menu(self, question, options, hint="↑/↓ to move · Enter to choose · Esc to go back"):
        s = self.style
        if not self.interactive:
            self.print(question, *[f"  {i}. {label}" for i, (label, _) in enumerate(options, 1)])
            raw = input("Number: ").strip()
            return int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(options) else None
        idx = 0
        column = max(16, max(len(f"{i + 1}. {label}") for i, (label, _) in enumerate(options)))
        while True:
            rows = [s(question, s.bold)]
            for i, (label, detail) in enumerate(options):
                chosen = i == idx
                pointer = s("❯", s.accent) if chosen else " "
                name = s(f"{i + 1}. {label}", s.accent) if chosen else f"{i + 1}. {label}"
                rows.append(f"{pointer} {pad(name, column)}  {s(detail, s.grey)}" if detail else f"{pointer} {name}")
            self.live(self.box(rows, colour=s.grey) + ["  " + s(hint, s.dim)])
            for key in self.keys():
                if key in ("up", "k"):
                    idx = (idx - 1) % len(options)
                elif key in ("down", "j"):
                    idx = (idx + 1) % len(options)
                elif key.isdigit() and 1 <= int(key) <= len(options):
                    idx = int(key) - 1
                elif key == "enter":
                    self.settle(s("❯ ", s.accent) + s(question + " ", s.grey) + options[idx][0])
                    return idx
                elif key in ("esc", "q"):
                    self.settle()
                    return None

    def ask_number(self, question, default, lo, hi, hint):
        s = self.style
        if not self.interactive:
            raw = input(f"{question} ({lo}-{hi}, Enter = {default}): ").strip()
            return default if not raw else (int(raw) if raw.isdigit() and lo <= int(raw) <= hi else None)
        text, error = "", ""
        while True:
            shown = text + s(" ", s.inverse) if text else s(" ", s.inverse) + s(str(default), s.dim)
            footer = s(error, s.red) if error else s(hint, s.dim)
            self.live([s(question, s.bold)] + self.box([s(">", s.accent) + " " + shown], colour=s.grey) + ["  " + footer])
            for key in self.keys():
                if key.isdigit() and len(text) < len(str(hi)):
                    text += key
                    error = ""
                elif key == "backspace":
                    text = text[:-1]
                    error = ""
                elif key == "esc":
                    self.settle()
                    return None
                elif key == "enter":
                    value = int(text) if text else default
                    if lo <= value <= hi:
                        self.settle(s("❯ ", s.accent) + s(question + " ", s.grey) + str(value))
                        return value
                    error = f"please type a number from {lo} to {hi}"

    def spin(self, task, status):
        s = self.style
        state = {"status": status, "result": None, "error": None}

        def run():
            try:
                state["result"] = task(lambda text: state.__setitem__("status", text))
            except BaseException as e:
                state["error"] = e

        worker = threading.Thread(target=run, daemon=True)
        start = time.time()
        worker.start()
        frame = 0
        while worker.is_alive():
            if self.interactive:
                glyph = FRAMES[frame % len(FRAMES)]
                self.write(f"\r\x1b[2K{s(glyph, s.accent)} {s(state['status'] + '…', s.accent)} "
                           f"{s(f'({time.time() - start:.0f}s)', s.dim)}")
            frame += 1
            worker.join(0.12)
        if self.interactive:
            self.write("\r\x1b[2K")
        if state["error"]:
            raise state["error"]
        return state["result"], time.time() - start
