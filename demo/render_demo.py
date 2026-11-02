"""Render the 2-minute ScamTeller demo video (1920x1080, 30 fps) from stored bench scores.

Nothing is inferred here: per-utterance risk scores come from the bench run files in
datasets/bench/ (the same files demo/replay_demo.py replays), transcripts and labels from the
eval split. Colors follow a validated dark-mode data-visualization palette: series slots 1-3 for
the three systems, reserved status colors (with icons) for alert states. Narration uses macOS
`say`; without --voice the video is silent.

Usage (from the repo root; needs pillow, numpy, imageio, imageio-ffmpeg):
  python demo/render_demo.py --out demo/ScamTeller_demo_2min.mp4 --voice "Ava (Premium)"
  python demo/render_demo.py ... --stills demo/stills/   # also write key frames as PNG
"""
import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SS = 1920, 1080, 30, 2  # SS = supersampling factor for the chart panel

# ---- dataviz reference palette, dark mode ----
PAGE = "#0d0d0d"
SURFACE = "#1a1a19"
INK = "#f0efec"
INK2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
BUBBLE_CALLER = "#262624"
BUBBLE_RECEIVER = "#21211f"
GOOD, SERIOUS, CRITICAL = "#0ca30c", "#ec835a", "#d03b3b"

FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
_fonts = {}


def font(size, weight="regular"):
    idx = {"regular": 0, "bold": 1, "light": 7, "medium": 10}[weight]
    key = (size, idx)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT, size, index=idx)
    return _fonts[key]


def rgb(h, a=255):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


def wrap(text, f, width, max_lines=None):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if f.getlength(t) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        while f.getlength(lines[-1] + " …") > width and " " in lines[-1]:
            lines[-1] = lines[-1].rsplit(" ", 1)[0]
        lines[-1] += " …"
    return lines


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


# ---- data -----------------------------------------------------------------------------
SYSTEMS = [
    ("ScamTeller", "fine-tuned 8B", "#3987e5", "ds3-eval-c2/scamteller.jsonl", 0.50),
    ("DeepSeek Flash", "closed model", "#d95926", "ds3-eval-c2/deepseek-flash.c2.jsonl", 0.85),
    ("Claude Haiku", "closed model", "#199e70", "ds3-eval-c2/claude-haiku.c2.jsonl", 0.90),
]
CALLS = [
    {"cid": "calib-101-00013", "tag": "CALL 1 OF 2  ·  SCAM", "title": "Fake parcel “handling fee”"},
    {"cid": "run-1101-00322", "tag": "CALL 2 OF 2  ·  LEGITIMATE (LOOKS LIKE A SCAM)",
     "title": "Real courier calling about import charges"},
]

# (utterance index revealed, seconds on screen, caption). Same index twice = hold, no reveal.
SCRIPT = {
    "calib-101-00013": [
        (0, 2.4, "A “claims assessor” calls about a parcel held at the receiver’s address."),
        (1, 0.9, "The receiver is wary, but suspicion alone isn’t evidence."),
        (2, 2.0, "The caller says a handling fee must be settled today, over the phone."),
        (3, 2.2, "Label: evidence is sufficient from here."),
        (3, 2.8, "ScamTeller alerts at utterance 3. Both closed models hold at 0.82, below their thresholds."),
        (4, 0.5, "The story keeps shifting. The closed models are still at 0.82."),
        (5, 0.6, "The story keeps shifting. The closed models are still at 0.82."),
        (6, 0.5, "The story keeps shifting. The closed models are still at 0.82."),
        (7, 2.2, "DeepSeek Flash and Claude Haiku alert at utterance 7: four utterances late."),
    ],
    "run-1101-00322": [
        (0, 2.2, "Call 2: a genuine courier calls about import charges on a parcel."),
        (1, 0.55, "The receiver is suspicious; the caller muddles “customs” and “courier”."),
        (2, 0.55, "The receiver is suspicious; the caller muddles “customs” and “courier”."),
        (3, 1.0, "The receiver is suspicious; the caller muddles “customs” and “courier”."),
        (4, 0.55, "The receiver is suspicious; the caller muddles “customs” and “courier”."),
        (5, 1.6, "No request for money or details. ScamTeller stays at 0.00."),
        (6, 0.45, "No request for money or details. ScamTeller stays at 0.00."),
        (7, 0.45, "No request for money or details. ScamTeller stays at 0.00."),
        (8, 0.45, "DeepSeek Flash peaks at 0.82, just under its 0.85 threshold."),
        (9, 1.4, "DeepSeek Flash peaks at 0.82, just under its 0.85 threshold."),
        (10, 0.45, "No request for money or details. ScamTeller stays at 0.00."),
        (11, 0.45, "No request for money or details. ScamTeller stays at 0.00."),
        (12, 0.45, "No request for money or details. ScamTeller stays at 0.00."),
        (13, 2.4, "Claude Haiku crosses 0.90: a false alarm on a legitimate call."),
    ],
}


def load(factory):
    split = os.path.join(factory, "datasets/eval/ds3-rubric-v3-detect07")
    convs = {json.loads(l)["conversation_id"]: json.loads(l) for l in open(f"{split}/conversations.jsonl")}
    metas = {json.loads(l)["conversation_id"]: json.loads(l) for l in open(f"{split}/metadata.jsonl")}
    runs = []
    for name, sub, color, rel, thr in SYSTEMS:
        by = {json.loads(l)["conversation_id"]: json.loads(l)
              for l in open(os.path.join(factory, "datasets/bench", rel))}
        runs.append(by)
    calls = []
    for c in CALLS:
        cid = c["cid"]
        utts = convs[cid]["utterances"]
        series = []
        for (name, sub, color, rel, thr), by in zip(SYSTEMS, runs):
            pts = sorted((s["index"], float(s["risk_score"])) for s in by[cid]["steps"]
                         if s.get("fresh", True))
            alarm = next((i for i, v in pts if v >= thr), None)
            series.append({"name": name, "sub": sub, "color": color, "thr": thr, "pts": pts, "alarm": alarm})
        calls.append({**c, "utts": utts, "det": metas[cid]["earliest_detection_index"],
                      "hard_negative": bool(metas[cid]["ground_truth"].get("hard_negative")),
                      "series": series})
    return calls


def value_at(pts, x):
    """Step value: latest evaluated score at or before utterance x (None before the first)."""
    v = None
    for i, s in pts:
        if i <= x + 1e-9:
            v = s
    return v


# ---- drawing primitives -----------------------------------------------------------------
def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def icon(d, kind, cx, cy, s, color):
    """Status icons drawn as vectors: check (good), clock (late), alert triangle (critical)."""
    if kind == "check":
        d.line([(cx - s * .45, cy), (cx - s * .1, cy + s * .35), (cx + s * .5, cy - s * .35)],
               fill=color, width=max(2, int(s * .18)), joint="curve")
    elif kind == "late":
        d.ellipse([cx - s * .5, cy - s * .5, cx + s * .5, cy + s * .5], outline=color, width=max(2, int(s * .14)))
        d.line([(cx, cy - s * .3), (cx, cy), (cx + s * .25, cy + s * .15)], fill=color, width=max(2, int(s * .14)))
    elif kind == "alert":
        d.polygon([(cx, cy - s * .55), (cx + s * .55, cy + s * .45), (cx - s * .55, cy + s * .45)], fill=color)
        d.line([(cx, cy - s * .2), (cx, cy + s * .12)], fill=INK, width=max(2, int(s * .12)))
        d.ellipse([cx - s * .06, cy + s * .22, cx + s * .06, cy + s * .34], fill=INK)


def chip(d, x, y, text, kind, color, f, align="left"):
    tw = f.getlength(text)
    h = f.size + 14
    w = tw + h + 18
    if align == "right":
        x -= w
    rrect(d, [x, y, x + w, y + h], h // 2, fill=color)
    icon(d, kind, x + h * .55, y + h / 2, h * .55, INK if kind != "alert" else INK)
    if kind == "alert":
        icon(d, "alert", x + h * .55, y + h / 2, h * .55, INK)
    d.text((x + h + 4, y + h / 2), text, font=f, fill=INK, anchor="lm")
    return w


def dashed_line(d, p0, p1, color, width, dash=10, gap=8):
    (x0, y0), (x1, y1) = p0, p1
    L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** .5
    n = int(L // (dash + gap)) + 1
    for k in range(n):
        a = k * (dash + gap) / L
        b = min(1.0, (k * (dash + gap) + dash) / L)
        d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
               fill=color, width=width)


# ---- panels -----------------------------------------------------------------------------
TX = (48, 124, 1000, 892)       # transcript panel
CH = (1040, 124, 1872, 892)     # chart panel


def draw_header(d, call):
    d.text((56, 60), "ScamTeller", font=font(46, "bold"), fill=INK, anchor="lm")
    d.text((56 + font(46, "bold").getlength("ScamTeller") + 18, 62), "real-time scam early warning",
           font=font(24, "light"), fill=INK2, anchor="lm")
    if call:
        f = font(19, "medium")
        tag = call["tag"]
        tw = f.getlength(tag)
        x1 = 1872
        rrect(d, [x1 - tw - 36, 36, x1, 84], 24, outline=rgb(INK2, 110), width=2)
        d.text((x1 - tw / 2 - 18, 60), tag, font=f, fill=INK2, anchor="mm")
        d.text((x1 - tw - 56, 60), call["title"], font=font(26, "medium"), fill=INK, anchor="rm")


def draw_transcript(img, call, k, t_reveal):
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0, x1, y1 = TX
    rrect(d, TX, 18, fill=SURFACE)
    d.text((x0 + 28, y0 + 30), "LIVE TRANSCRIPT", font=font(17, "medium"), fill=MUTED, anchor="lm")
    ftxt, flab = font(23), font(17, "medium")
    maxw = 690
    blocks = []
    for i in range(k + 1):
        u = call["utts"][i]
        lines = wrap(u["text"], ftxt, maxw - 44, max_lines=4)
        bh = 22 + len(lines) * 31 + 16
        blocks.append((i, u, lines, bh))
    # stack upward from the bottom; newest slides in
    y = y1 - 24
    top_limit = y0 + 58
    slide = (1 - ease(t_reveal)) * 26
    for i, u, lines, bh in reversed(blocks):
        newest = i == k
        oy = slide if newest else 0
        alpha = int(255 * (ease(t_reveal) if newest else 1))
        by1 = y + oy
        by0 = by1 - bh
        lab_y = by0 - 26
        if lab_y < top_limit:
            break
        caller = u["speaker"] == "caller"
        bw = max(ftxt.getlength(l) for l in lines) + 44
        bx0 = x0 + 28 if caller else x1 - 28 - bw
        fill = rgb(BUBBLE_CALLER, alpha) if caller else rgb(BUBBLE_RECEIVER, alpha)
        is_det = call["det"] is not None and i == call["det"]
        outline = rgb(INK2, alpha) if is_det else rgb("#ffffff", int(alpha * .10))
        rrect(d, [bx0, by0, bx0 + bw, by1], 16, fill=fill, outline=outline, width=2 if is_det else 1)
        label = f"{'Caller' if caller else 'Receiver'}  ·  utterance {i}"
        if is_det:
            label += "   —   LABEL: evidence sufficient from here"
        d.text((bx0 + (4 if caller else bw - 4), lab_y + 10), label, font=flab,
               fill=rgb(INK2 if is_det else MUTED, alpha), anchor="lm" if caller else "rm")
        for j, line in enumerate(lines):
            d.text((bx0 + 22, by0 + 22 + j * 31), line, font=ftxt, fill=rgb(INK if newest else INK2, alpha),
                   anchor="lt")
        y = lab_y - 14


def draw_chart(img, call, xpos, k):
    """xpos: float utterance position the lines are drawn up to (animated)."""
    x0, y0, x1, y1 = CH
    w, h = (x1 - x0) * SS, (y1 - y0) * SS
    layer = Image.new("RGBA", (w, h), rgb(PAGE))
    d = ImageDraw.Draw(layer, "RGBA")
    S = lambda v: v * SS
    rrect(d, [0, 0, w - 1, h - 1], S(18), fill=SURFACE)
    d.text((S(28), S(30)), "SCAM RISK AFTER EACH UTTERANCE", font=font(S(17), "medium"), fill=MUTED, anchor="lm")
    n = len(call["utts"])
    px0, px1 = S(84), w - S(36)
    xs = lambda i: px0 + (px1 - px0) * i / (n - 1)
    lane_top, lane_h, plot_h = S(66), S(222), S(120)
    for li, s in enumerate(call["series"]):
        ly = lane_top + li * lane_h
        # header: swatch + name + threshold, current score on the right
        d.ellipse([S(28), ly + S(14), S(44), ly + S(30)], fill=s["color"])
        d.text((S(54), ly + S(22)), s["name"], font=font(S(25), "medium"), fill=INK, anchor="lm")
        nx = S(54) + font(S(25), "medium").getlength(s["name"]) + S(12)
        d.text((nx, ly + S(23)), f'{s["sub"]}  ·  threshold {s["thr"]:.2f}', font=font(S(18)), fill=MUTED, anchor="lm")
        cur = value_at(s["pts"], min(xpos, k))
        score_w = font(S(34), "bold").getlength("0.00")
        if cur is not None:
            d.text((w - S(36), ly + S(22)), f"{cur:.2f}", font=font(S(34), "bold"), fill=INK, anchor="rm")
        if s["alarm"] is not None and s["alarm"] <= xpos + 1e-9:
            txt = (f'FALSE ALARM · utt {s["alarm"]}' if call["det"] is None
                   else f'ALERT · utterance {s["alarm"]}')
            chip(d, w - S(36) - score_w - S(18), ly + S(6), txt, "alert", CRITICAL, font(S(17), "medium"),
                 align="right")
        # plot area
        py0 = ly + S(56)
        py1 = py0 + plot_h
        yv = lambda v: py1 - (py1 - py0) * v
        for gv in (0.0, 0.5, 1.0):
            d.line([(px0, yv(gv)), (px1, yv(gv))], fill=GRID if gv else AXIS, width=S(1))
            d.text((px0 - S(14), yv(gv)), f"{gv:g}", font=font(S(15)), fill=MUTED, anchor="rm")
        dashed_line(d, (px0, yv(s["thr"])), (px1, yv(s["thr"])), MUTED, S(2), dash=S(9), gap=S(7))
        # step line up to xpos
        pts = [(i, v) for i, v in s["pts"] if i <= xpos + 1e-9]
        if pts:
            path = []
            for j, (i, v) in enumerate(pts):
                if j:
                    path.append((xs(i), yv(pts[j - 1][1])))
                path.append((xs(i), yv(v)))
            path.append((xs(xpos), yv(pts[-1][1])))
            d.line(path, fill=s["color"], width=S(3), joint="curve")
            for i, v in pts:
                r = S(5)
                d.ellipse([xs(i) - r - S(2), yv(v) - r - S(2), xs(i) + r + S(2), yv(v) + r + S(2)], fill=SURFACE)
                d.ellipse([xs(i) - r, yv(v) - r, xs(i) + r, yv(v) + r], fill=s["color"])
        # ground-truth line, inside this lane's plot only
        if call["det"] is not None and call["det"] <= xpos + 1e-9:
            gx = xs(call["det"])
            dashed_line(d, (gx, py0 - S(6)), (gx, py1), INK2, S(2), dash=S(6), gap=S(6))
        # alert marker: a ring around the first point at or above the threshold
        if s["alarm"] is not None and s["alarm"] <= xpos + 1e-9:
            ax, av = xs(s["alarm"]), yv(dict(s["pts"])[s["alarm"]])
            r = S(11)
            d.ellipse([ax - r, av - r, ax + r, av + r], outline=CRITICAL, width=S(3))
    # x axis; the labelled utterance is marked under its tick
    ay = lane_top + 3 * lane_h - S(18)
    for i in range(n):
        is_det = call["det"] is not None and i == call["det"] and i <= xpos + 1e-9
        d.text((xs(i), ay), str(i), font=font(S(15), "bold" if is_det else "regular"),
               fill=INK if is_det else (INK2 if i <= k else MUTED), anchor="mm")
        if is_det:
            d.text((xs(i), ay + S(24)), "label", font=font(S(15), "medium"), fill=INK2, anchor="mm")
    d.text((px1, ay + S(24)), "utterance", font=font(S(15)), fill=MUTED, anchor="rm")
    img.paste(layer.resize((x1 - x0, y1 - y0), Image.LANCZOS), (x0, y0))


def draw_caption(d, text, alpha):
    if not text:
        return
    f = font(31)
    lines = wrap(text, f, 1640, max_lines=2)
    y = 966 if len(lines) == 1 else 948
    for j, line in enumerate(lines):
        d.text((W / 2, y + j * 40), line, font=f, fill=rgb(INK, int(255 * alpha)), anchor="mm")


def draw_footer(d):
    d.text((W / 2, 1052),
           "Stored scores replayed on the eval split  ·  ScamTeller at 0.50, untuned  ·  closed models at thresholds "
           "tuned on 100 training calls  ·  synthetic calls (scam-call-factory ds3)",
           font=font(16), fill=MUTED, anchor="mm")


def call_frame(call, k, t_reveal, caption, cap_alpha, overlay=None):
    img = Image.new("RGBA", (W, H), rgb(PAGE))
    d = ImageDraw.Draw(img, "RGBA")
    draw_header(d, call)
    draw_transcript(img, call, k, t_reveal)
    prev = k - 1 if t_reveal < 1 and k > 0 else k
    xpos = prev + (k - prev) * ease(t_reveal)
    draw_chart(img, call, xpos, k)
    d = ImageDraw.Draw(img, "RGBA")
    draw_caption(d, caption, cap_alpha)
    draw_footer(d)
    if overlay:
        overlay(img)
    return img.convert("RGB")


# ---- full-screen cards ------------------------------------------------------------------
def card_intro(alpha):
    img = Image.new("RGBA", (W, H), rgb(PAGE))
    d = ImageDraw.Draw(img, "RGBA")
    a = lambda c: rgb(c, int(255 * alpha))
    d.text((W / 2, 400), "ScamTeller", font=font(120, "bold"), fill=a(INK), anchor="mm")
    d.text((W / 2, 492), "Real-time phone-scam early warning", font=font(40, "light"), fill=a(INK2), anchor="mm")
    for j, line in enumerate(["Telling someone it was a scam after they’ve read out the code is worthless.",
                              "The question isn’t “is this a scam?”  It’s “how early can you tell?”"]):
        d.text((W / 2, 610 + j * 50), line, font=font(31), fill=a(INK), anchor="mm")
    return img.convert("RGB")


def card_transition(alpha, call):
    img = Image.new("RGBA", (W, H), rgb(PAGE))
    d = ImageDraw.Draw(img, "RGBA")
    a = lambda c: rgb(c, int(255 * alpha))
    d.text((W / 2, 470), call["tag"], font=font(26, "medium"), fill=a(INK2), anchor="mm")
    d.text((W / 2, 540), call["title"], font=font(56, "bold"), fill=a(INK), anchor="mm")
    sub = ("Same topic, same pressure points, an equally suspicious receiver. But the caller is genuine."
           if call["det"] is None else "")
    if sub:
        d.text((W / 2, 620), sub, font=font(30), fill=a(INK2), anchor="mm")
    return img.convert("RGB")


def result_overlay(call, alpha):
    def apply(img):
        ov = Image.new("RGBA", (W, H), (0, 0, 0, int(190 * alpha)))
        img.alpha_composite(ov)
        d = ImageDraw.Draw(img, "RGBA")
        cw, chh = 1100, 430
        cx0, cy0 = (W - cw) / 2, (H - chh) / 2 - 30
        rrect(d, [cx0, cy0, cx0 + cw, cy0 + chh], 22, fill=rgb(SURFACE, int(255 * alpha)),
              outline=rgb(AXIS, int(255 * alpha)), width=2)
        a = lambda c: rgb(c, int(255 * alpha))
        title = ("When did each detector alert?" if call["det"] is not None
                 else "Did anyone raise a false alarm?")
        d.text((cx0 + 48, cy0 + 60), title, font=font(38, "bold"), fill=a(INK), anchor="lm")
        if call["det"] is not None:
            d.text((cx0 + 48, cy0 + 108), f"Label: evidence sufficient from utterance {call['det']}",
                   font=font(22), fill=a(INK2), anchor="lm")
        else:
            d.text((cx0 + 48, cy0 + 108), "Label: legitimate call, evidence never sufficient",
                   font=font(22), fill=a(INK2), anchor="lm")
        for r, s in enumerate(call["series"]):
            ry = cy0 + 180 + r * 78
            d.ellipse([cx0 + 50, ry - 9, cx0 + 68, ry + 9], fill=a(s["color"]))
            d.text((cx0 + 86, ry), s["name"], font=font(30, "medium"), fill=a(INK), anchor="lm")
            peak = max(v for _, v in s["pts"])
            if call["det"] is not None:
                if s["alarm"] is None:
                    what, kind, col = "never alerted", "alert", CRITICAL
                else:
                    lag = s["alarm"] - call["det"]
                    what = f"alerted at utterance {s['alarm']}"
                    kind, col, tag = (("check", GOOD, "on the label") if lag == 0 else
                                      ("late", SERIOUS, f"{lag} late") if lag > 0 else
                                      ("late", SERIOUS, f"{-lag} early"))
                    what = f"{what}"
            else:
                if s["alarm"] is None:
                    what, kind, col, tag = f"no alert  ·  peak {peak:.2f}", "check", GOOD, "correct"
                else:
                    what, kind, col, tag = f"alerted at utterance {s['alarm']}", "alert", CRITICAL, "false alarm"
            d.text((cx0 + 430, ry), what, font=font(28), fill=a(INK2), anchor="lm")
            if alpha > 0.6:
                chip(d, cx0 + cw - 48, ry - 22, tag, kind, col, font(22, "medium"), align="right")
    return apply


# Benchmark numbers for the video. Hosted detectors: docs/BENCH_REPORT.md Table C (test PR-AUC)
# and §6 (median / p95 latency, concurrency 4). ScamTeller and the untuned base: bench-table on the
# test split; latency from the original training run's offline eval (p50 70.7 / 69.7 ms, p95 123.5 /
# 122.6 ms; batch 1, no cache; scamteller.latency measures the same condition).
BENCH_TEST = [  # name, median ms, p95 ms, test PR-AUC
    ("ScamTeller", 71, 124, 0.938),
    ("Qwen3-8B, not fine-tuned", 70, 123, 0.770),   # same base, same prompt, frozen 
    ("Jev", 90, 161, 0.806),
    ("Claude Haiku", 1102, 1397, 0.832),
    ("DeepSeek Flash", 1109, 1421, 0.898),
    ("GPT Luna", 1429, 1802, 0.789),
    ("Claude Opus", 1995, 5526, 0.893),
    ("DeepSeek Pro", 2006, 2641, 0.784),
    ("GPT Sol", 2396, 3368, 0.888),
]
LABEL_OFFSET = {  # (dx, dy, anchor) in px, placed by hand to avoid collisions
    "ScamTeller": (18, -18, "lb"), "Qwen3-8B, not fine-tuned": (20, 0, "lm"), "Jev": (16, 22, "lt"), "Claude Haiku": (-14, -14, "rb"),
    "DeepSeek Flash": (-14, -14, "rb"), "GPT Luna": (-14, 18, "rt"), "Claude Opus": (0, -22, "mb"),
    "DeepSeek Pro": (14, 18, "lt"), "GPT Sol": (14, 16, "lt"),
}


def _card(alpha):
    img = Image.new("RGBA", (W, H), rgb(PAGE))
    return img, ImageDraw.Draw(img, "RGBA"), (lambda c, f=1.0: rgb(c, int(255 * alpha * f)))


def card_how(alpha):
    img, d, a = _card(alpha)
    d.text((W / 2, 150), "How ScamTeller works", font=font(58, "bold"), fill=a(INK), anchor="mm")
    # utterance timeline: grey before the labelled moment, blue (alerting) from it on
    x0, x1, y = 300, 1620, 360
    d.line([(x0, y), (x1, y)], fill=a(GRID), width=4)
    n, det = 16, 3
    for i in range(n):
        x = x0 + (x1 - x0) * i / (n - 1)
        col = "#3987e5" if i >= det else AXIS
        d.ellipse([x - 13, y - 13, x + 13, y + 13], fill=a(col))
    xd = x0 + (x1 - x0) * det / (n - 1)
    chip(d, xd - 95, y - 92, "ALERT HERE", "alert", CRITICAL, font(22, "bold")) if alpha > 0.5 else None
    d.line([(xd, y - 40), (xd, y - 18)], fill=a(INK2), width=3)
    d.text((xd, y + 46), "label: evidence sufficient", font=font(22, "medium"), fill=a(INK2), anchor="mt")
    d.text((x1, y + 46), "utterance →".replace(" →", ""), font=font(20), fill=a(MUTED), anchor="rt")
    steps = [("1", "Score the call after every utterance", "The transcript so far goes in; one verdict token comes out."),
             ("2", "The label is a moment", "The first utterance where a careful listener has grounds to stop cooperating."),
             ("3", "Fine-tune on those moments", "Qwen3-8B with LoRA, trained on synthetic calls from scam-call-factory.")]
    for j, (num, head, body) in enumerate(steps):
        x = 180 + j * 540
        rrect(d, [x, 520, x + 500, 800], 22, fill=a(SURFACE))
        d.ellipse([x + 36, 556, x + 86, 606], fill=a("#3987e5"))
        d.text((x + 61, 581), num, font=font(28, "bold"), fill=a(INK), anchor="mm")
        for k, line in enumerate(wrap(head, font(30, "bold"), 420)):
            d.text((x + 40, 640 + k * 38), line, font=font(30, "bold"), fill=a(INK), anchor="lt")
        hl = len(wrap(head, font(30, "bold"), 420))
        for k, line in enumerate(wrap(body, font(24), 420)):
            d.text((x + 40, 650 + hl * 38 + k * 32), line, font=font(24), fill=a(INK2), anchor="lt")
    return img.convert("RGB")


def card_fair(alpha):
    img, d, a = _card(alpha)
    d.text((W / 2, 120), "How thresholds were set", font=font(56, "bold"), fill=a(INK), anchor="mm")
    # data flow: which split is used for what
    boxes = [("TRAIN  ·  777 calls  ·  5,933 decision points", "ScamTeller is fine-tuned here.", "Hosted thresholds: tuned on 100 of these."),
             ("EVAL  ·  119 calls  ·  912 decision points", "Same scam families, new calls.", "Reported as “in domain”."),
             ("TEST  ·  104 calls  ·  817 decision points", "6 scam families never seen in training.", "Reported as “unseen families”.")]
    bw, bh, gap, by = 530, 170, 50, 200
    bx0 = (W - (3 * bw + 2 * gap)) / 2
    for i, (head, l1, l2) in enumerate(boxes):
        x = bx0 + i * (bw + gap)
        rrect(d, [x, by, x + bw, by + bh], 20, fill=a(SURFACE), outline=a(AXIS), width=2)
        d.text((x + 28, by + 42), head, font=font(22, "bold"), fill=a(INK), anchor="lm")
        d.text((x + 28, by + 92), l1, font=font(23), fill=a(INK2), anchor="lm")
        d.text((x + 28, by + 130), l2, font=font(23), fill=a(INK2), anchor="lm")
        if i < 2:
            ax0, ax1, ay = x + bw + 12, x + bw + gap - 12, by + bh / 2
            d.line([(ax0, ay), (ax1 - 12, ay)], fill=a(MUTED), width=3)
            d.polygon([(ax1, ay), (ax1 - 16, ay - 9), (ax1 - 16, ay + 9)], fill=a(MUTED))
    d.text((W / 2, by + bh + 34), "A decision point is a moment in a call where every detector gives a score and the label says scam or not yet.",
           font=font(21), fill=a(MUTED), anchor="mm")
    d.text((W / 2, by + bh + 72), "Each detector’s score has its own scale, so each gets its own threshold. All were fixed "
           "before eval or test was scored; test is used only for final numbers.", font=font(23), fill=a(INK2), anchor="mm")
    rows = [("7 hosted detectors (6 LLMs + Jev): tuned on 100 training calls.",
             "Objective: catch the most scams with ≤ 10% false alarms on look-alike legitimate calls."),
            ("ScamTeller: fixed at 0.50, untuned — those 100 calls were in its training data.",
             "The same tuning would pick 0.30 and catch more (93% vs 86% on unseen families)."),
            ("The ranking that follows uses PR-AUC, which needs no threshold at all.",
             "So the headline comparison does not depend on any of these choices.")]
    for j, (head, body) in enumerate(rows):
        y = 520 + j * 156
        rrect(d, [240, y, 1680, y + 130], 20, fill=a(SURFACE))
        d.rounded_rectangle([240, y, 252, y + 130], radius=6, fill=a("#3987e5" if j == 1 else AXIS))
        d.text((290, y + 46), head, font=font(30, "medium"), fill=a(INK), anchor="lm")
        d.text((290, y + 90), body, font=font(24), fill=a(INK2), anchor="lm")
    return img.convert("RGB")


def card_bench(alpha):
    """Held-out families: threshold-free ranking quality (PR-AUC) vs latency (log x). ScamTeller is the one
    highlighted series; every other point is muted and direct-labelled, so identity is never colour-only."""
    img, d, a = _card(alpha)
    d.text((110, 92), "On six scam families never seen in training", font=font(46, "bold"), fill=a(INK), anchor="lm")
    d.text((110, 146), "Seven hosted detectors (incl. flagships), the untuned Qwen3-8B, and ScamTeller  ·  ranking quality vs. speed",
           font=font(26, "light"), fill=a(INK2), anchor="lm")
    px0, px1, py0, py1 = 260, 1760, 230, 880
    rrect(d, [150, 196, 1830, 940], 22, fill=a(SURFACE))
    import math
    lx = lambda ms: px0 + (px1 - px0) * (math.log10(ms) - math.log10(40)) / (math.log10(8000) - math.log10(40))
    ly = lambda v: py1 - (py1 - py0) * (v - 0.76) / (0.96 - 0.76)
    for v in (0.80, 0.85, 0.90, 0.95):
        d.line([(px0, ly(v)), (px1, ly(v))], fill=a(GRID), width=2)
        d.text((px0 - 16, ly(v)), f"{v:.2f}", font=font(20), fill=a(MUTED), anchor="rm")
    for ms in (50, 100, 300, 1000, 3000):
        d.line([(lx(ms), py1), (lx(ms), py1 + 8)], fill=a(AXIS), width=2)
        d.text((lx(ms), py1 + 20), f"{ms:,} ms" if ms < 1000 else f"{ms // 1000} s", font=font(20), fill=a(MUTED), anchor="mt")
    d.line([(px0, py1), (px1, py1)], fill=a(AXIS), width=2)
    d.text(((px0 + px1) / 2, py1 + 52), "latency per utterance  (dot = median, line = to p95; log scale)",
           font=font(20), fill=a(MUTED), anchor="mt")
    d.text((px0 - 16, py0 - 26), "test PR-AUC (threshold-free)", font=font(20), fill=a(MUTED), anchor="lm")
    dashed_line(d, (lx(200), py0), (lx(200), py1), MUTED, 2, dash=10, gap=8)
    d.text((lx(200) + 10, py0 + 6), "real-time target ≈ 200 ms", font=font(20), fill=a(MUTED), anchor="lt")
    base = next(b for b in BENCH_TEST if b[0].startswith("Qwen3-8B"))
    top = next(b for b in BENCH_TEST if b[0] == "ScamTeller")
    ax = lx(base[1]) - 34
    d.line([(ax, ly(base[3]) - 16), (ax, ly(top[3]) + 26)], fill=a("#3987e5", 0.9), width=4)
    d.polygon([(ax, ly(top[3]) + 12), (ax - 11, ly(top[3]) + 30), (ax + 11, ly(top[3]) + 30)], fill=a("#3987e5", 0.9))
    d.text((ax - 14, (ly(base[3]) + ly(top[3])) / 2), "fine-tuning", font=font(21, "medium"), fill=a(INK2), anchor="rm")
    d.text((ax - 14, (ly(base[3]) + ly(top[3])) / 2 + 28), f"+{top[3] - base[3]:.3f}", font=font(21, "bold"),
           fill=a(INK), anchor="rm")
    for name, med, p95, auc in BENCH_TEST:
        if name == base[0]:
            x, y = lx(med), ly(auc)
            d.line([(x, y), (lx(p95), y)], fill=a("#3987e5", 0.5), width=3)
            d.ellipse([x - 13, y - 13, x + 13, y + 13], fill=a(SURFACE), outline=a("#3987e5"), width=4)
            dx, dy, anc = LABEL_OFFSET[name]
            pos, txt, f = (lx(p95) + dx, y + dy), f"{name}  {auc:.3f}", font(22)
            bx = d.textbbox(pos, txt, font=f, anchor=anc)
            d.rectangle([bx[0] - 6, bx[1] - 4, bx[2] + 6, bx[3] + 4], fill=a(SURFACE))
            d.text(pos, txt, font=f, fill=a(INK2), anchor=anc)
            continue
        hi = name == "ScamTeller"
        col = "#3987e5" if hi else MUTED
        x, y = lx(med), ly(auc)
        d.line([(x, y), (lx(p95), y)], fill=a(col, 0.8), width=4 if hi else 3)
        r = 15 if hi else 10
        d.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3], fill=a(SURFACE))
        d.ellipse([x - r, y - r, x + r, y + r], fill=a(col))
        dx, dy, anc = LABEL_OFFSET[name]
        d.text((x + dx, y + dy), f"{name}  {auc:.3f}" if hi else name, font=font(30 if hi else 22, "bold" if hi else "regular"),
               fill=a(INK if hi else INK2), anchor=anc)
    d.text((W / 2, 1010), "Hosted APIs measured at concurrency 4 (upper bounds); ScamTeller and untuned Qwen3-8B: offline forward pass, "
           "batch 1, no cache, same prompt.  Source: scam-call-factory BENCH_REPORT (2026-11-02).", font=font(18), fill=a(MUTED), anchor="mm")
    return img.convert("RGB")


def card_stats(alpha):
    img, d, a = _card(alpha)
    d.text((W / 2, 170), "In domain (eval: 119 calls, 912 decision points)", font=font(48, "bold"), fill=a(INK), anchor="mm")
    tiles = [("0 / 57", "false alarms on", "legitimate calls"),
             ("0.63", "pinpointing the moment of evidence", "(pair accuracy; next best 0.49)"),
             ("0.05–0.95", "every threshold in this range keeps", "false alarms ≤ 10% (next best 0.85–0.95)")]
    tw, gap = 520, 40
    tx0 = (W - (3 * tw + 2 * gap)) / 2
    for i, (big, l1, l2) in enumerate(tiles):
        x = tx0 + i * (tw + gap)
        rrect(d, [x, 290, x + tw, 640], 22, fill=a(SURFACE))
        d.text((x + tw / 2, 420), big, font=font(92 if len(big) < 7 else 76, "bold"), fill=a(INK), anchor="mm")
        d.text((x + tw / 2, 540), l1, font=font(25), fill=a(INK2), anchor="mm")
        d.text((x + tw / 2, 578), l2, font=font(23), fill=a(MUTED), anchor="mm")
    d.text((W / 2, 740), "Pair accuracy: separating the utterance before the labelled moment from the one at it. "
           "Thresholds cannot buy it.", font=font(24), fill=a(INK2), anchor="mm")
    return img.convert("RGB")


def card_limits(alpha):
    img, d, a = _card(alpha)
    d.text((W / 2, 200), "Where it falls short", font=font(58, "bold"), fill=a(INK), anchor="mm")
    rows = [("Best ranking is not full recall: at its untuned 0.50, ScamTeller catches 86% of unseen-family scams.",
             "That threshold favours fewer false alarms (1 / 52). Tuned like the others (0.30): 93%. Claude Opus: 100%, at 2 s."),
            ("5 of its 6 misses show their evidence only in the final utterance.",
             "Late reveals are the next thing to fix."),
            ("Synthetic calls, labels from an LLM annotator.",
             "Human adjudication of the labels is under way.")]
    for j, (head, body) in enumerate(rows):
        y = 340 + j * 170
        rrect(d, [150, y, 1770, y + 140], 20, fill=a(SURFACE))
        icon(d, "late", 210, y + 70, 40, a(SERIOUS))
        d.text((270, y + 50), head, font=font(29, "medium"), fill=a(INK), anchor="lm")
        d.text((270, y + 96), body, font=font(24), fill=a(INK2), anchor="lm")
    return img.convert("RGB")


def card_outro(alpha):
    img, d, a = _card(alpha)
    d.text((W / 2, 440), "The label is a moment, not a class.", font=font(68, "bold"), fill=a(INK), anchor="mm")
    d.text((W / 2, 540), "ScamTeller  ·  real-time phone-scam early warning", font=font(32, "light"), fill=a(INK2), anchor="mm")
    d.text((W / 2, 900), "Data & benchmark: scam-call-factory   ·   Model: Qwen3-8B per-prefix fine-tune   ·   "
           "Narration: AI-generated voice", font=font(20), fill=a(MUTED), anchor="mm")
    return img.convert("RGB")


# ---- narration ------------------------------------------------------------------------
# Spoken lines, anchored to segments of the timeline. Numbers are spelled the way they should
# be read aloud; on-screen captions keep the digits.
NARRATION = {
    "intro": "A scam warning that arrives after the victim reads out the code is worthless. "
             "The question is how early you can tell.",
    "how": "ScamTeller scores a call after every utterance, with a single token. We fine-tuned an "
           "eight-billion-parameter model on the moment a careful listener has enough evidence to stop cooperating.",
    ("calib-101-00013", 0): "A scam call about a held parcel. Three detectors score it live: ScamTeller, and two closed models.",
    ("calib-101-00013", 2): "The caller wants a handling fee paid today, over the phone.",
    ("calib-101-00013", 3): "This is where the evidence becomes sufficient.",
    ("calib-101-00013", 4): "ScamTeller alerts right here. The closed models stall below their thresholds.",
    ("calib-101-00013", 8): "They alert four utterances later.",
    ("trans", 1): "Now, a genuine courier.",
    ("run-1101-00322", 0): "Same topic, an equally suspicious receiver, but no request for money or details.",
    ("run-1101-00322", 5): "ScamTeller stays at zero.",
    ("run-1101-00322", 13): "Claude Haiku raises a false alarm.",
    "bench": "On scam families never seen in training, ScamTeller ranks highest by threshold-free PR-AUC, up from "
             "zero point seven seven before fine-tuning, at about seventy milliseconds. Claude Opus and GPT Sol are "
             "excellent judges, but at two seconds per utterance they are too slow for a live call.",
    "fair": "A note on fairness. Each detector gets its own threshold, fixed before any test call was scored. "
            "The hosted models were tuned on training calls; ScamTeller stays at an untuned zero point five.",
    "stats": "In domain: zero false alarms on fifty-seven legitimate calls, and the best at pinpointing when "
             "evidence appears.",
    "limits": "It's not done. Best ranking isn't full recall: at its untuned threshold, ScamTeller catches eighty-six "
              "percent of unseen-family scams, ninety-three if tuned, where Claude Opus catches all of them. Most misses "
              "come in the final utterance. And our labels come from an LLM, on synthetic calls.",
    "outro": "The label is a moment, not a class.",
}
SR = 22050          # narration sample rate
LEAD, TAIL = 0.2, 0.3   # seconds of silence before / after a line within its segment group


def synthesize(voice, rate, workdir):
    """Render every narration line with macOS `say`; return {key: int16 samples}."""
    import subprocess
    import wave
    os.makedirs(workdir, exist_ok=True)
    clips = {}
    for i, (key, text) in enumerate(NARRATION.items()):
        path = os.path.join(workdir, f"line{i:02d}.wav")
        subprocess.run(["say", "-v", voice, "-r", str(rate), "--file-format=WAVE",
                        f"--data-format=LEI16@{SR}", "-o", path, text], check=True)
        with wave.open(path) as w:
            assert w.getframerate() == SR and w.getnchannels() == 1
            clips[key] = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return clips


# ---- timeline ---------------------------------------------------------------------------
def fade(t, dur, f_in=0.4, f_out=0.4):
    return min(1.0, t / f_in if f_in else 1, (dur - t) / f_out if f_out else 1)


def segments(calls):
    """Ordered segments: (kind, payload, seconds, narration key or None)."""
    segs = [("intro", None, 6.0, "intro"), ("card", "how", 10.0, "how")]
    for ci, call in enumerate(calls):
        if ci:
            segs.append(("trans", ci, 1.6, ("trans", ci)))
        prev_k = -1
        for ei, (k, dur, cap) in enumerate(SCRIPT[call["cid"]]):
            key = (call["cid"], ei)
            segs.append(("event", (ci, k, cap, k != prev_k), dur, key if key in NARRATION else None))
            prev_k = k
        segs.append(("result", ci, 2.2, None))
    segs += [("card", "fair", 11.5, "fair"), ("card", "bench", 14.0, "bench"), ("card", "stats", 7.0, "stats"),
             ("card", "limits", 9.0, "limits"), ("outro", None, 3.0, "outro")]
    return segs


def fit_to_narration(segs, clips):
    """Stretch each narration group (a narrated segment plus the unnarrated ones after it, within the same
    kind) so the line fits with LEAD/TAIL padding; return new segs and line start times (seconds)."""
    segs = [list(s) for s in segs]
    i = 0
    while i < len(segs):
        key = segs[i][3]
        j = i + 1
        while j < len(segs) and segs[j][3] is None and segs[j][0] == segs[i][0]:
            j += 1
        if key is not None and key in clips:
            need = LEAD + len(clips[key]) / SR + TAIL
            have = sum(s[2] for s in segs[i:j])
            if need > have:
                segs[j - 1][2] += need - have
        i = j
    starts, t = {}, 0.0
    for s in segs:
        if s[3] is not None:
            starts[s[3]] = t + LEAD
        t += s[2]
    return [tuple(s) for s in segs], starts, t


CARDS = {"how": card_how, "fair": card_fair, "bench": card_bench, "stats": card_stats, "limits": card_limits}


def frames(calls, segs):
    prev_cap, last = {}, {}
    for kind, payload, dur, _ in segs:
        n = int(round(dur * FPS))
        if kind in ("intro", "outro", "card", "trans"):
            for i in range(n):
                al = round(fade(i / FPS, dur, 0.4, 0.4), 2)
                yield (kind, payload, al)
        elif kind == "event":
            ci, k, cap, reveal = payload
            new_cap = cap != prev_cap.get(ci)
            for i in range(n):
                t = i / FPS
                tr = round(min(1.0, t / 0.35), 3) if reveal else 1.0
                ca = round(min(1.0, t / 0.25), 2) if new_cap else 1.0
                yield ("call", ci, k, tr, cap, ca, 0.0)
            prev_cap[ci] = cap
            last[ci] = (k, cap)
        elif kind == "result":
            k, cap = last[payload]
            for i in range(n):
                yield ("call", payload, k, 1.0, cap, 1.0, round(min(1.0, (i / FPS) / 0.35), 2))


def render(key, calls):
    kind = key[0]
    if kind == "intro":
        return card_intro(key[2])
    if kind == "outro":
        return card_outro(key[2])
    if kind == "card":
        return CARDS[key[1]](key[2])
    if kind == "trans":
        return card_transition(key[2], calls[key[1]])
    _, ci, k, tr, cap, ca, ov = key
    call = calls[ci]
    return call_frame(call, k, tr, cap, ca, result_overlay(call, ov) if ov > 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default=".", help="repo root (holds datasets/)")
    ap.add_argument("--out", default="ScamTeller_demo_2min.mp4")
    ap.add_argument("--stills", default=None, help="directory for a few key frames (PNG)")
    ap.add_argument("--voice", default=None, help='macOS voice for narration, e.g. "Ava (Premium)"; omit = silent')
    ap.add_argument("--rate", type=int, default=192, help="narration words per minute")
    args = ap.parse_args()
    calls = load(args.factory)

    if args.stills:
        os.makedirs(args.stills, exist_ok=True)
        c1, c2 = calls
        cap3 = SCRIPT[c1["cid"]][4][2]
        render(("call", 0, 3, 1.0, cap3, 1.0, 0.0), calls).save(f"{args.stills}/call1_u3.png")
        render(("call", 0, 7, 1.0, SCRIPT[c1["cid"]][8][2], 1.0, 0.0), calls).save(f"{args.stills}/call1_u7.png")
        render(("call", 0, 15, 1.0, "", 0, 1.0), calls).save(f"{args.stills}/call1_result.png")
        render(("call", 1, 13, 1.0, SCRIPT[c2["cid"]][-1][2], 1.0, 0.0), calls).save(f"{args.stills}/call2_u13.png")
        render(("call", 1, 13, 1.0, "", 0, 1.0), calls).save(f"{args.stills}/call2_result.png")
        card_intro(1.0).save(f"{args.stills}/intro.png")
        card_outro(1.0).save(f"{args.stills}/outro.png")
        for name, fn in CARDS.items():
            fn(1.0).save(f"{args.stills}/card_{name}.png")
        print(f"stills -> {args.stills}")

    segs = segments(calls)
    clips = synthesize(args.voice, args.rate, os.path.join(os.path.dirname(os.path.abspath(args.out)), "voice"))\
        if args.voice else {}
    segs, starts, total = fit_to_narration(segs, clips)

    silent = args.out if not args.voice else args.out.replace(".mp4", ".silent.mp4")
    writer = imageio.get_writer(silent, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p",
                                macro_block_size=8)
    last_key, last_arr, n = None, None, 0
    for key in frames(calls, segs):
        if key != last_key:
            last_arr = np.asarray(render(key, calls))
            last_key = key
        writer.append_data(last_arr)
        n += 1
    writer.close()
    print(f"{n} frames ({n / FPS:.1f} s) -> {silent}")

    if args.voice:
        import subprocess
        import wave
        import imageio_ffmpeg
        track = np.zeros(int((n / FPS + 1) * SR), dtype=np.int32)
        for key, clip in clips.items():
            s = int(starts[key] * SR)
            track[s:s + len(clip)] += clip
        track = np.clip(track, -32768, 32767).astype(np.int16)
        wav = args.out.replace(".mp4", ".narration.wav")
        with wave.open(wav, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(track.tobytes())
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", silent, "-i", wav,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", args.out], check=True)
        print(f"narrated ({len(clips)} lines, voice {args.voice!r}, {args.rate} wpm) -> {args.out}")


if __name__ == "__main__":
    main()
