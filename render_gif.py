#!/usr/bin/env python3
"""
Render the SUBLEQ terminal animation as an animated GIF using Pillow.

Usage:
    python3 render_gif.py

Output:
    subleq_demo.gif (in the same directory)
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WIDTH = 900
HEIGHT = 600
BG_COLOR = "#1a1b26"
FONT_SIZE = 15
LEFT_MARGIN = 30
TOP_MARGIN = 20
LINE_HEIGHT = 20  # pixels per line

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subleq_demo.gif")

# Color scheme
C_DEFAULT  = "#c0caf5"
C_GREEN    = "#9ece6a"
C_CYAN     = "#7dcfff"
C_YELLOW   = "#e0af68"
C_RED      = "#f7768e"
C_MAGENTA  = "#bb9af7"
C_ORANGE   = "#ff9e64"
C_DIM      = "#565f89"
C_WHITE    = "#ffffff"

# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------
def load_font(size):
    """Try to load a monospace font, falling back to default."""
    candidates = [
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/Library/Fonts/Courier New.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                print(f"  Using font: {path}")
                return font
            except Exception:
                continue
    print("  Using Pillow default font (no monospace found)")
    return ImageFont.load_default()

FONT = load_font(FONT_SIZE)
FONT_SMALL = load_font(FONT_SIZE - 2)

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def new_frame():
    """Create a new blank frame."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    return img, draw


def draw_text(draw, x, y, text, color=C_DEFAULT):
    """Draw text at pixel position."""
    draw.text((x, y), text, fill=color, font=FONT)


def draw_text_small(draw, x, y, text, color=C_DEFAULT):
    """Draw text at pixel position with smaller font."""
    draw.text((x, y), text, fill=color, font=FONT_SMALL)


def text_width(text):
    """Get the pixel width of text using the current font."""
    bbox = FONT.getbbox(text)
    return bbox[2] - bbox[0]


def text_height(text="X"):
    """Get the pixel height of text using the current font."""
    bbox = FONT.getbbox(text)
    return bbox[3] - bbox[1]


def draw_lines(draw, lines, start_y=TOP_MARGIN, x=LEFT_MARGIN):
    """
    Draw a list of (text, color) or [(text, color), ...] lines.
    Each element in `lines` can be:
      - A tuple (text, color) for a single-color line
      - A list of (text, color) tuples for mixed-color spans on one line
    Returns the y position after the last line.
    """
    y = start_y
    for line in lines:
        if isinstance(line, list):
            # Multiple spans on one line
            cx = x
            for text, color in line:
                draw_text(draw, cx, y, text, color)
                cx += text_width(text)
        elif isinstance(line, tuple):
            text, color = line
            draw_text(draw, x, y, text, color)
        else:
            # Plain string
            draw_text(draw, x, y, str(line), C_DEFAULT)
        y += LINE_HEIGHT
    return y


def draw_cursor(draw):
    """Draw a subtle blinking cursor at bottom-right."""
    draw_text(draw, WIDTH - 40, HEIGHT - 30, "\u258c", C_DIM)


def center_x(text):
    """Get x position to center text horizontally."""
    w = text_width(text)
    return max(0, (WIDTH - w) // 2)


def center_block_y(num_lines):
    """Get y position to vertically center a block of lines."""
    block_h = num_lines * LINE_HEIGHT
    return max(0, (HEIGHT - block_h) // 2)


def progress_bar(filled_frac, bar_len=30):
    """Return (filled_str, empty_str) for a text progress bar."""
    filled = int(filled_frac * bar_len)
    filled = min(filled, bar_len)
    return "\u2588" * filled, "\u2591" * (bar_len - filled)


# ---------------------------------------------------------------------------
# Frame rendering functions
# ---------------------------------------------------------------------------

def render_frame_1():
    """Title frame."""
    img, draw = new_frame()

    box_w = 57
    top_line    = "\u2554" + "\u2550" * box_w + "\u2557"
    bottom_line = "\u255a" + "\u2550" * box_w + "\u255d"
    title    = "A Transformer That Learned to Be a Computer"
    subtitle = "SUBLEQ: One instruction. Turing complete."

    # Center vertically
    cy = center_block_y(4)
    cx = center_x(top_line)

    draw_text(draw, cx, cy, top_line, C_CYAN)
    # Title line
    draw_text(draw, cx, cy + LINE_HEIGHT, "\u2551", C_CYAN)
    draw_text(draw, cx + text_width("\u2551  "), cy + LINE_HEIGHT, title, C_WHITE)
    draw_text(draw, cx + text_width("\u2551  " + title + " " * (box_w - 2 - len(title))), cy + LINE_HEIGHT, "\u2551", C_CYAN)

    # Subtitle line
    draw_text(draw, cx, cy + LINE_HEIGHT * 2, "\u2551", C_CYAN)
    draw_text(draw, cx + text_width("\u2551  "), cy + LINE_HEIGHT * 2, subtitle, C_CYAN)
    draw_text(draw, cx + text_width("\u2551  " + subtitle + " " * (box_w - 2 - len(subtitle))), cy + LINE_HEIGHT * 2, "\u2551", C_CYAN)

    draw_text(draw, cx, cy + LINE_HEIGHT * 3, bottom_line, C_CYAN)

    draw_cursor(draw)
    return img


def render_frame_2():
    """The Instruction frame."""
    img, draw = new_frame()

    lines = [
        ("", C_DEFAULT),
        ("  THE INSTRUCTION", C_CYAN),
        ("", C_DEFAULT),
        ("  SUBLEQ a b c", C_WHITE),
        ("", C_DEFAULT),
        [("     mem[b] ", C_DEFAULT), ("-=", C_YELLOW), (" mem[a]", C_DEFAULT)],
        [("     if mem[b] ", C_DEFAULT), ("\u2264 0", C_YELLOW), (": goto c", C_DEFAULT)],
        ("", C_DEFAULT),
        ("  That's it. One instruction. Turing complete.", C_ORANGE),
    ]

    draw_lines(draw, lines, start_y=center_block_y(len(lines)))
    draw_cursor(draw)
    return img


def render_frame_3():
    """Loading Program frame."""
    img, draw = new_frame()

    lines = [
        ("", C_DEFAULT),
        [("  LOADING PROGRAM: ", C_CYAN), ("multiply(7, 9)", C_GREEN)],
        ("", C_DEFAULT),
        [("    Instr 0:  ", C_DIM), ("SUBLEQ 24 26  3", C_MAGENTA), ("   // result += 7", C_DIM)],
        [("    Instr 1:  ", C_DIM), ("SUBLEQ 27 25 -1", C_MAGENTA), ("   // counter--; halt if done", C_DIM)],
        [("    Instr 2:  ", C_DIM), ("SUBLEQ  9  9  0", C_MAGENTA), ("   // unconditional jump -> 0", C_DIM)],
        ("", C_DEFAULT),
        ("  Data cells:", C_CYAN),
        [("    mem[24] = ", C_CYAN), (" -7", C_WHITE), ("   (negated multiplicand)", C_DIM)],
        [("    mem[25] = ", C_CYAN), ("  9", C_WHITE), ("   (counter / multiplier)", C_DIM)],
        [("    mem[26] = ", C_CYAN), ("  0", C_WHITE), ("   (result accumulator)", C_DIM)],
        [("    mem[27] = ", C_CYAN), ("  1", C_WHITE), ("   (constant 1)", C_DIM)],
    ]

    draw_lines(draw, lines, start_y=60)
    draw_cursor(draw)
    return img


def render_frame_4():
    """Initial Memory Grid frame."""
    img, draw = new_frame()

    y = 30
    draw_text(draw, LEFT_MARGIN, y, "  Initial memory state:", C_CYAN)
    draw_text(draw, WIDTH - 150, y, "PC = 0", C_RED)
    y += LINE_HEIGHT * 2

    # Memory values
    mem = [
        24, 26,  3,  27, 25, -1,  9,  9,   # row 0: code
         0,  0,  0,   0,  0,  0,  0,  0,   # row 1
         0,  0,  0,   0,  0,  0,  0,  0,   # row 2
        -7,  9,  0,   1,  0,  0,  0,  0,   # row 3: data
    ]

    cw = 5  # cell width in chars
    indent = "     "

    # Column headers
    header = indent
    for col in range(8):
        header += f"{col:>{cw}}"
    draw_text(draw, LEFT_MARGIN, y, header, C_DIM)
    y += LINE_HEIGHT

    # Grid rows
    def draw_grid_border(y_pos, style="mid"):
        if style == "top":
            line = indent + "\u250c" + ("\u2500" * cw + "\u252c") * 7 + "\u2500" * cw + "\u2510"
        elif style == "mid":
            line = indent + "\u251c" + ("\u2500" * cw + "\u253c") * 7 + "\u2500" * cw + "\u2524"
        else:
            line = indent + "\u2514" + ("\u2500" * cw + "\u2534") * 7 + "\u2500" * cw + "\u2518"
        draw_text(draw, LEFT_MARGIN, y_pos, line, C_DIM)

    for row in range(4):
        # Top/mid border
        if row == 0:
            draw_grid_border(y, "top")
        else:
            draw_grid_border(y, "mid")
        y += LINE_HEIGHT

        # Data cells
        cx = LEFT_MARGIN + text_width(indent + "\u2502")
        draw_text(draw, LEFT_MARGIN + text_width(indent), y, "\u2502", C_DIM)

        for col in range(8):
            addr = row * 8 + col
            val = mem[addr]
            vs = f"{val:>{cw}}"

            # Color logic
            if row == 0 and addr < 3:
                # PC = 0 cells
                color = C_RED
            elif row == 0 and val != 0:
                color = C_MAGENTA
            elif row == 3 and (addr == 24 or addr == 25 or addr == 26 or addr == 27):
                color = C_CYAN
            elif val == 0:
                color = C_DIM
            else:
                color = C_DEFAULT

            draw_text(draw, cx, y, vs, color)
            cx += text_width(vs)
            if col < 7:
                draw_text(draw, cx, y, "\u2502", C_DIM)
                cx += text_width("\u2502")

        draw_text(draw, cx, y, "\u2502", C_DIM)

        # Row annotations
        if row == 0:
            draw_text(draw, cx + text_width("\u2502 "), y, " <- code", C_MAGENTA)
        elif row == 3:
            draw_text(draw, cx + text_width("\u2502 "), y, " <- data", C_CYAN)

        y += LINE_HEIGHT

    # Bottom border
    draw_grid_border(y, "bottom")
    y += LINE_HEIGHT

    # Bottom address labels (row 3)
    addr_line = indent
    for col in range(8):
        addr_line += f"{24 + col:>{cw}}"
    draw_text(draw, LEFT_MARGIN, y, addr_line, C_DIM)

    draw_cursor(draw)
    return img


def render_iteration_frame(iteration, result_before, result_after,
                           counter_before, counter_after, step_base):
    """Render an iteration frame (frames 5-7)."""
    img, draw = new_frame()

    y = 30
    draw_text(draw, LEFT_MARGIN, y, f"  --- Iteration {iteration}/9 ", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width(f"  --- Iteration {iteration}/9 "), y, "-" * 35, C_DIM)
    y += LINE_HEIGHT * 2

    # Step 0: SUBLEQ(24, 26, 3) - result += 7
    step_num = step_base
    draw_text(draw, LEFT_MARGIN, y, f"  Step {step_num}", C_DIM)
    draw_text(draw, LEFT_MARGIN + text_width(f"  Step {step_num} "), y, "|", C_DIM)
    draw_text(draw, LEFT_MARGIN + text_width(f"  Step {step_num} | "), y, "SUBLEQ(24, 26, 3)", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width(f"  Step {step_num} | SUBLEQ(24, 26, 3)     "), y, "result += 7", C_DIM)
    y += LINE_HEIGHT

    draw_text(draw, LEFT_MARGIN, y, f"         |", C_DIM)
    spans = [
        (f" mem[26] = {result_before} - (-7) = ", C_DEFAULT),
        (f"{result_after}", C_YELLOW),
        ("           > fall through", C_DIM),
    ]
    cx = LEFT_MARGIN + text_width(f"         | ")
    for text, color in spans:
        draw_text(draw, cx, y, text, color)
        cx += text_width(text)
    y += LINE_HEIGHT

    # Brain emoji line for step 0
    draw_text(draw, LEFT_MARGIN + text_width("         | "), y, "Transformer ->", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width("         | Transformer -> "), y, "matches interpreter", C_GREEN)
    y += LINE_HEIGHT * 2

    # Step 1: SUBLEQ(27, 25, -1) - counter--
    step_num = step_base + 1
    draw_text(draw, LEFT_MARGIN, y, f"  Step {step_num}", C_DIM)
    draw_text(draw, LEFT_MARGIN + text_width(f"  Step {step_num} "), y, "|", C_DIM)
    draw_text(draw, LEFT_MARGIN + text_width(f"  Step {step_num} | "), y, "SUBLEQ(27, 25, -1)", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width(f"  Step {step_num} | SUBLEQ(27, 25, -1)    "), y, "counter--", C_DIM)
    y += LINE_HEIGHT

    draw_text(draw, LEFT_MARGIN, y, f"         |", C_DIM)
    spans = [
        (f" mem[25] = {counter_before} - 1 = ", C_DEFAULT),
        (f"{counter_after}", C_YELLOW),
        ("              > fall through", C_DIM),
    ]
    cx = LEFT_MARGIN + text_width(f"         | ")
    for text, color in spans:
        draw_text(draw, cx, y, text, color)
        cx += text_width(text)
    y += LINE_HEIGHT

    draw_text(draw, LEFT_MARGIN + text_width("         | "), y, "Transformer ->", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width("         | Transformer -> "), y, "matches interpreter", C_GREEN)
    y += LINE_HEIGHT * 2

    # Step 2: SUBLEQ(9, 9, 0) - jump -> 0
    step_num = step_base + 2
    draw_text(draw, LEFT_MARGIN, y, f"  Step {step_num}", C_DIM)
    draw_text(draw, LEFT_MARGIN + text_width(f"  Step {step_num} "), y, "|", C_DIM)
    draw_text(draw, LEFT_MARGIN + text_width(f"  Step {step_num} | "), y, "SUBLEQ(9, 9, 0)", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width(f"  Step {step_num} | SUBLEQ(9, 9, 0)       "), y, "jump -> 0", C_DIM)
    y += LINE_HEIGHT

    draw_text(draw, LEFT_MARGIN, y, f"         |", C_DIM)
    spans = [
        (f" mem[9] = 0 - 0 = ", C_DEFAULT),
        ("0", C_YELLOW),
        ("                  > branch -> 0", C_DIM),
    ]
    cx = LEFT_MARGIN + text_width(f"         | ")
    for text, color in spans:
        draw_text(draw, cx, y, text, color)
        cx += text_width(text)
    y += LINE_HEIGHT

    draw_text(draw, LEFT_MARGIN + text_width("         | "), y, "Transformer ->", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width("         | Transformer -> "), y, "matches interpreter", C_GREEN)
    y += LINE_HEIGHT * 2

    # Progress bar
    frac = result_after / 63.0
    filled, empty = progress_bar(frac)
    draw_text(draw, LEFT_MARGIN, y, f"  result: [", C_DIM)
    cx = LEFT_MARGIN + text_width(f"  result: [")
    draw_text(draw, cx, y, filled, C_GREEN)
    cx += text_width(filled)
    draw_text(draw, cx, y, empty, C_DIM)
    cx += text_width(empty)
    draw_text(draw, cx, y, f"]  {result_after}/63", C_DIM)

    draw_cursor(draw)
    return img


def render_frame_8():
    """Fast forward frame (iterations 4-8)."""
    img, draw = new_frame()

    y = 30
    draw_text(draw, LEFT_MARGIN, y, "  --- Iterations 4-8 ", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width("  --- Iterations 4-8 "), y, "-" * 33, C_DIM)
    y += LINE_HEIGHT * 2

    draw_text(draw, LEFT_MARGIN + 60, y, ">> accelerating...", C_DIM)
    y += LINE_HEIGHT * 2

    iters = [
        (4, 28, 5),
        (5, 35, 4),
        (6, 42, 3),
        (7, 49, 2),
        (8, 56, 1),
    ]
    for it, res, cnt in iters:
        draw_text(draw, LEFT_MARGIN + 30, y, f"Iteration {it}:  ", C_DIM)
        cx = LEFT_MARGIN + 30 + text_width(f"Iteration {it}:  ")
        draw_text(draw, cx, y, f"result = {res}", C_YELLOW)
        cx += text_width(f"result = {res}  ")
        draw_text(draw, cx, y, f"counter = {cnt}", C_YELLOW)
        cx += text_width(f"counter = {cnt}  ")
        draw_text(draw, cx, y, "OK", C_GREEN)
        y += LINE_HEIGHT

    y += LINE_HEIGHT

    # Progress bar
    frac = 56.0 / 63.0
    filled, empty = progress_bar(frac)
    draw_text(draw, LEFT_MARGIN, y, f"  result: [", C_DIM)
    cx = LEFT_MARGIN + text_width(f"  result: [")
    draw_text(draw, cx, y, filled, C_GREEN)
    cx += text_width(filled)
    draw_text(draw, cx, y, empty, C_DIM)
    cx += text_width(empty)
    draw_text(draw, cx, y, f"]  56/63", C_DIM)
    y += LINE_HEIGHT * 2

    draw_text(draw, LEFT_MARGIN, y, "  Every step verified against ground-truth interpreter.", C_DIM)

    draw_cursor(draw)
    return img


def render_frame_9():
    """Final iteration frame."""
    img, draw = new_frame()

    y = 30
    draw_text(draw, LEFT_MARGIN, y, "  --- Iteration 9/9 (final) ", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width("  --- Iteration 9/9 (final) "), y, "-" * 26, C_DIM)
    y += LINE_HEIGHT * 2

    # Step 24: SUBLEQ(24, 26, 3) - result += 7
    draw_text(draw, LEFT_MARGIN, y, "  Step 24", C_DIM)
    draw_text(draw, LEFT_MARGIN + text_width("  Step 24 "), y, "|", C_DIM)
    draw_text(draw, LEFT_MARGIN + text_width("  Step 24 | "), y, "SUBLEQ(24, 26, 3)", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width("  Step 24 | SUBLEQ(24, 26, 3)    "), y, "result += 7", C_DIM)
    y += LINE_HEIGHT

    draw_text(draw, LEFT_MARGIN + text_width("          | "), y, "mem[26] = 56 - (-7) = ", C_DEFAULT)
    draw_text(draw, LEFT_MARGIN + text_width("          | mem[26] = 56 - (-7) = "), y, "63", C_YELLOW)
    draw_text(draw, LEFT_MARGIN + text_width("          | mem[26] = 56 - (-7) = 63        "), y, "> fall through", C_DIM)
    y += LINE_HEIGHT

    draw_text(draw, LEFT_MARGIN + text_width("          | "), y, "Transformer ->", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width("          | Transformer -> "), y, "CORRECT", C_GREEN)
    y += LINE_HEIGHT * 2

    # Step 25: SUBLEQ(27, 25, -1) - counter--
    draw_text(draw, LEFT_MARGIN, y, "  Step 25", C_DIM)
    draw_text(draw, LEFT_MARGIN + text_width("  Step 25 "), y, "|", C_DIM)
    draw_text(draw, LEFT_MARGIN + text_width("  Step 25 | "), y, "SUBLEQ(27, 25, -1)", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width("  Step 25 | SUBLEQ(27, 25, -1)   "), y, "counter--", C_DIM)
    y += LINE_HEIGHT

    draw_text(draw, LEFT_MARGIN + text_width("          | "), y, "mem[25] = 1 - 1 = ", C_DEFAULT)
    draw_text(draw, LEFT_MARGIN + text_width("          | mem[25] = 1 - 1 = "), y, "0", C_YELLOW)
    draw_text(draw, LEFT_MARGIN + text_width("          | mem[25] = 1 - 1 = 0             "), y, "> 0 <= 0 -> ", C_DIM)
    draw_text(draw, LEFT_MARGIN + text_width("          | mem[25] = 1 - 1 = 0             > 0 <= 0 -> "), y, "HALT!", C_RED)
    y += LINE_HEIGHT

    draw_text(draw, LEFT_MARGIN + text_width("          | "), y, "Transformer ->", C_CYAN)
    draw_text(draw, LEFT_MARGIN + text_width("          | Transformer -> "), y, "CORRECT", C_GREEN)
    y += LINE_HEIGHT * 2

    # Full progress bar
    filled, _ = progress_bar(1.0)
    draw_text(draw, LEFT_MARGIN, y, f"  result: [", C_DIM)
    cx = LEFT_MARGIN + text_width(f"  result: [")
    draw_text(draw, cx, y, filled, C_GREEN)
    cx += text_width(filled)
    draw_text(draw, cx, y, f"]  63/63  ", C_DIM)
    cx += text_width(f"]  63/63  ")
    draw_text(draw, cx, y, "DONE", C_GREEN)
    y += LINE_HEIGHT * 2

    draw_text(draw, LEFT_MARGIN, y, "  | PROGRAM HALTED after 26 steps", C_YELLOW)

    draw_cursor(draw)
    return img


def render_frame_10():
    """Result reveal frame."""
    img, draw = new_frame()

    box_lines = [
        "+-------------------------------------------+",
        "|                                           |",
        "|    mem[26] = 63                            |",
        "|                                           |",
        "|    7  x  9  =  63     CORRECT             |",
        "|                                           |",
        "|    Never seen during training.             |",
        "|    Learned from random single steps.       |",
        "|    Executed 26 steps perfectly.            |",
        "|                                           |",
        "+-------------------------------------------+",
    ]

    # Calculate center position
    cy = center_block_y(len(box_lines))
    cx_base = center_x(box_lines[0])

    for i, line in enumerate(box_lines):
        yy = cy + i * LINE_HEIGHT
        if i == 0 or i == len(box_lines) - 1:
            # Border lines
            draw_text(draw, cx_base, yy, line, C_GREEN)
        elif i == 1 or i == 3 or i == 5 or i == 9:
            # Empty border lines
            draw_text(draw, cx_base, yy, "|", C_GREEN)
            draw_text(draw, cx_base + text_width("|" + " " * 43), yy, "|", C_GREEN)
        elif i == 2:
            # mem[26] = 63
            draw_text(draw, cx_base, yy, "|", C_GREEN)
            draw_text(draw, cx_base + text_width("|    "), yy, "mem[26] = ", C_DEFAULT)
            draw_text(draw, cx_base + text_width("|    mem[26] = "), yy, "63", C_GREEN)
            draw_text(draw, cx_base + text_width("|" + " " * 43), yy, "|", C_GREEN)
        elif i == 4:
            # 7 x 9 = 63  CORRECT
            draw_text(draw, cx_base, yy, "|", C_GREEN)
            draw_text(draw, cx_base + text_width("|    "), yy, "7  x  9  =  ", C_WHITE)
            draw_text(draw, cx_base + text_width("|    7  x  9  =  "), yy, "63", C_GREEN)
            draw_text(draw, cx_base + text_width("|    7  x  9  =  63     "), yy, "CORRECT", C_GREEN)
            draw_text(draw, cx_base + text_width("|" + " " * 43), yy, "|", C_GREEN)
        elif i in (6, 7, 8):
            # Explanation lines
            texts = {
                6: "Never seen during training.",
                7: "Learned from random single steps.",
                8: "Executed 26 steps perfectly.",
            }
            draw_text(draw, cx_base, yy, "|", C_GREEN)
            draw_text(draw, cx_base + text_width("|    "), yy, texts[i], C_CYAN)
            draw_text(draw, cx_base + text_width("|" + " " * 43), yy, "|", C_GREEN)

    draw_cursor(draw)
    return img


def render_frame_11():
    """Emergent programs frame."""
    img, draw = new_frame()

    y = 40
    draw_text(draw, LEFT_MARGIN, y, "  EMERGENT PROGRAMS", C_CYAN)
    y += LINE_HEIGHT
    draw_text(draw, LEFT_MARGIN, y, "  Programs the model was never trained on:", C_DIM)
    y += LINE_HEIGHT * 2

    programs = [
        ("multiply(7, 9)", "63", "26 steps"),
        ("fibonacci(5)",   "55", "39 steps"),
        ("div(100, 7)",    "14", "71 steps"),
        ("isqrt(81)",       "9", "55 steps"),
    ]

    for name, result, steps in programs:
        draw_text(draw, LEFT_MARGIN + 40, y, f"{name:>16s}", C_WHITE)
        cx = LEFT_MARGIN + 40 + text_width(f"{name:>16s}")
        draw_text(draw, cx, y, "  --->  ", C_DIM)
        cx += text_width("  --->  ")
        draw_text(draw, cx, y, f"{result:>3s}", C_GREEN)
        cx += text_width(f"{result:>3s}  ")
        draw_text(draw, cx, y, "CORRECT", C_GREEN)
        cx += text_width("CORRECT  ")
        draw_text(draw, cx, y, f"({steps})", C_DIM)
        y += LINE_HEIGHT

    y += LINE_HEIGHT

    # Separator
    draw_text(draw, LEFT_MARGIN, y, "  " + "\u2501" * 51, C_DIM)
    y += LINE_HEIGHT * 2

    draw_text(draw, LEFT_MARGIN, y, "  141/141 multiplication table entries: ", C_DEFAULT)
    draw_text(draw, LEFT_MARGIN + text_width("  141/141 multiplication table entries: "), y, "ALL CORRECT", C_GREEN)
    y += LINE_HEIGHT
    draw_text(draw, LEFT_MARGIN, y, "  Test accuracy on held-out programs: ", C_DEFAULT)
    draw_text(draw, LEFT_MARGIN + text_width("  Test accuracy on held-out programs: "), y, "100%", C_GREEN)

    draw_cursor(draw)
    return img


def render_frame_12():
    """Closing frame."""
    img, draw = new_frame()

    y = 60
    stats = [
        ("4.9M parameters", C_WHITE),
        ("32 memory cells", C_WHITE),
        ("8-bit integers", C_WHITE),
        ("One instruction: subtract and branch", C_WHITE),
    ]
    for text, color in stats:
        draw_text(draw, LEFT_MARGIN + 30, y, text, color)
        y += LINE_HEIGHT

    y += LINE_HEIGHT
    draw_text(draw, LEFT_MARGIN + 30, y, "Trained on random single steps.", C_CYAN)
    y += LINE_HEIGHT
    draw_text(draw, LEFT_MARGIN + 30, y, "Never saw a full program.", C_CYAN)
    y += LINE_HEIGHT * 2

    draw_text(draw, LEFT_MARGIN + 30, y, "Emerged: a general-purpose computer.", C_GREEN)
    y += LINE_HEIGHT * 2

    # Architecture info
    draw_text(draw, LEFT_MARGIN + 30, y, "-" * 45, C_DIM)
    y += LINE_HEIGHT
    draw_text(draw, LEFT_MARGIN + 30, y, "Pre-LN Transformer  |  256-dim  |  6 layers  |  8 heads", C_DIM)
    y += LINE_HEIGHT
    draw_text(draw, LEFT_MARGIN + 30, y, "Trained on Apple M1  |  ~2 hours  |  from scratch", C_DIM)
    y += LINE_HEIGHT
    draw_text(draw, LEFT_MARGIN + 30, y, "-" * 45, C_DIM)

    draw_cursor(draw)
    return img


# ---------------------------------------------------------------------------
# Main: assemble and save GIF
# ---------------------------------------------------------------------------

def main():
    print("Rendering SUBLEQ demo GIF...")
    print()

    frames = []
    durations = []

    renderers = [
        ("Frame  1/12: Title",              render_frame_1,   2500),
        ("Frame  2/12: The Instruction",    render_frame_2,   3000),
        ("Frame  3/12: Loading Program",    render_frame_3,   3000),
        ("Frame  4/12: Initial Memory",     render_frame_4,   2500),
        ("Frame  5/12: Iteration 1",        lambda: render_iteration_frame(1, 0, 7, 9, 8, 0),    1500),
        ("Frame  6/12: Iteration 2",        lambda: render_iteration_frame(2, 7, 14, 8, 7, 3),   1500),
        ("Frame  7/12: Iteration 3",        lambda: render_iteration_frame(3, 14, 21, 7, 6, 6),  1500),
        ("Frame  8/12: Fast Forward",       render_frame_8,   1500),
        ("Frame  9/12: Final Iteration",    render_frame_9,   2000),
        ("Frame 10/12: Result Reveal",      render_frame_10,  3000),
        ("Frame 11/12: Emergent Programs",  render_frame_11,  3500),
        ("Frame 12/12: Closing",            render_frame_12,  4000),
    ]

    for label, renderer, duration in renderers:
        print(f"  Rendering {label}...")
        img = renderer()
        # Convert to palette mode for smaller GIF
        img_p = img.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
        frames.append(img_p)
        durations.append(duration)

    print()
    print(f"  Saving GIF to {OUTPUT_PATH}...")

    frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=durations,
        optimize=True,
    )

    file_size = os.path.getsize(OUTPUT_PATH)
    size_kb = file_size / 1024
    size_mb = file_size / (1024 * 1024)

    print()
    if size_mb >= 1:
        print(f"  File size: {size_mb:.2f} MB")
    else:
        print(f"  File size: {size_kb:.1f} KB")
    print(f"  Saved to:  {OUTPUT_PATH}")
    print(f"  Frames:    {len(frames)}")
    print(f"  Total duration: {sum(durations)/1000:.1f}s")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
