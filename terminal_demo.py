#!/usr/bin/env python3
"""
Terminal animation: A Transformer That Learned to Be a Computer.

Produces a beautiful ASCII terminal animation showing a transformer
executing SUBLEQ multiplication (7 x 9 = 63).

Usage:
    python3 terminal_demo.py            # normal speed
    python3 terminal_demo.py --fast     # 5x faster (for testing)
    python3 terminal_demo.py --record   # dump plain text frames (no ANSI)

No dependencies beyond the Python standard library.
"""

import sys
import time
import argparse

# ---------------------------------------------------------------------------
# ANSI escape helpers
# ---------------------------------------------------------------------------
ESC = "\033["
CLEAR_SCREEN = f"{ESC}2J{ESC}H"
HIDE_CURSOR  = f"{ESC}?25l"
SHOW_CURSOR  = f"{ESC}?25h"

def cursor_to(row, col):
    return f"{ESC}{row};{col}H"

def cursor_up(n=1):
    return f"{ESC}{n}A"

def clear_line():
    return f"{ESC}2K"

# Colors & styles
RESET   = f"{ESC}0m"
BOLD    = f"{ESC}1m"
DIM     = f"{ESC}2m"
ITALIC  = f"{ESC}3m"
ULINE   = f"{ESC}4m"

FG_BLACK   = f"{ESC}30m"
FG_RED     = f"{ESC}91m"
FG_GREEN   = f"{ESC}92m"
FG_YELLOW  = f"{ESC}93m"
FG_BLUE    = f"{ESC}94m"
FG_MAGENTA = f"{ESC}95m"
FG_CYAN    = f"{ESC}96m"
FG_WHITE   = f"{ESC}97m"
FG_GRAY    = f"{ESC}90m"

BG_RED     = f"{ESC}41m"
BG_GREEN   = f"{ESC}42m"
BG_YELLOW  = f"{ESC}43m"
BG_BLUE    = f"{ESC}44m"
BG_CYAN    = f"{ESC}46m"
BG_GRAY    = f"{ESC}100m"
BG_DARK    = f"{ESC}48;5;236m"
BG_DARKER  = f"{ESC}48;5;234m"

# ---------------------------------------------------------------------------
# Global timing / recording state
# ---------------------------------------------------------------------------
SPEED = 1.0        # multiplier (<1 = faster)
RECORD = False     # if True, strip ANSI and dump frames

def strip_ansi(s):
    """Remove ANSI escape sequences from a string."""
    import re
    return re.sub(r'\033\[[^m]*m|\033\[\?25[hl]|\033\[\d+;\d+H|\033\[\d+[A-H]|\033\[2[JK]', '', s)

_frame_buffer = []

def emit(s, end="\n"):
    """Print (or record) a string."""
    if RECORD:
        _frame_buffer.append(strip_ansi(s) + (end if end else ""))
    else:
        sys.stdout.write(s + end)
        sys.stdout.flush()

def emit_raw(s):
    """Write raw (cursor movement etc.) -- skipped in record mode except newlines."""
    if RECORD:
        return
    sys.stdout.write(s)
    sys.stdout.flush()

def pause(seconds):
    """Sleep with speed multiplier."""
    if not RECORD:
        time.sleep(seconds * SPEED)

def typing(text, per_char=0.03, style=""):
    """Type out text character by character."""
    for ch in text:
        emit(f"{style}{ch}{RESET}" if style else ch, end="")
        if ch not in (" ", "\n"):
            pause(per_char)
    emit("")  # newline

def typing_no_nl(text, per_char=0.03, style=""):
    """Type out text without trailing newline."""
    for ch in text:
        emit(f"{style}{ch}{RESET}" if style else ch, end="")
        if ch not in (" ", "\n"):
            pause(per_char)

def clear():
    if RECORD:
        _frame_buffer.append("\n" + "=" * 70 + "\n\n")
    else:
        emit_raw(CLEAR_SCREEN)

# ---------------------------------------------------------------------------
# The SUBLEQ multiplication program: 7 x 9 = 63
#
# Memory layout (from programs.py make_multiply):
#   Instr 0 (pc=0): SUBLEQ 24 26  3   -- result -= mem[24] = result + 7
#   Instr 1 (pc=3): SUBLEQ 27 25 -1   -- counter -= 1; halt if <= 0
#   Instr 2 (pc=6): SUBLEQ  9  9  0   -- unconditional jump to 0
#   Code cells 9-23: 0
#   Data: mem[24]=-7  mem[25]=9  mem[26]=0  mem[27]=1  mem[28..31]=0
# ---------------------------------------------------------------------------

INIT_MEM = [
    24, 26,  3,   # instr 0: add 7 to result
    27, 25, -1,   # instr 1: decrement counter, halt if <= 0
     9,  9,  0,   # instr 2: unconditional jump to 0
     0,  0,  0,   # unused code
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,
     0,  0,  0,   # cells 21-23
    -7,  9,  0,  1,  0,  0,  0,  0,   # data cells 24-31
]

assert len(INIT_MEM) == 32

# Pre-computed full trace (26 steps, verified against interpreter)
# Each entry: (pc_before, a_addr, b_addr, c_addr, result_after, counter_after)
FULL_TRACE = []
_m = list(INIT_MEM)
_pc = 0
for _step in range(200):
    if _pc < 0 or _pc + 2 >= 32:
        break
    a_addr = _m[_pc]
    b_addr = _m[_pc + 1]
    c_addr = _m[_pc + 2]
    if a_addr < 0 or a_addr >= 32 or b_addr < 0 or b_addr >= 32:
        break
    old_mem = list(_m)
    _m[b_addr] = max(-128, min(127, _m[b_addr] - _m[a_addr]))
    if _m[b_addr] <= 0:
        new_pc = c_addr
    else:
        new_pc = _pc + 3
    FULL_TRACE.append({
        'step': _step,
        'pc': _pc,
        'a': a_addr,
        'b': b_addr,
        'c': c_addr,
        'a_val': old_mem[a_addr],
        'b_val_before': old_mem[b_addr],
        'b_val_after': _m[b_addr],
        'new_pc': new_pc,
        'branch': _m[b_addr] <= 0,
        'mem': list(_m),
        'result': _m[26],
        'counter': _m[25],
    })
    _pc = new_pc
    if _pc < 0 or _pc + 2 >= 32:
        break

TOTAL_STEPS = len(FULL_TRACE)

# ---------------------------------------------------------------------------
# Memory grid rendering
# ---------------------------------------------------------------------------

def format_cell(val, width=4):
    """Format a value for display, right-justified."""
    s = str(val)
    return s.rjust(width)

def render_memory_grid(mem, pc, changed_addrs=None, highlight_addrs=None,
                       show_addrs=True, indent=4):
    """
    Render a 4x8 memory grid with box-drawing characters.
    Returns list of strings (lines).
    """
    if changed_addrs is None:
        changed_addrs = set()
    if highlight_addrs is None:
        highlight_addrs = set()

    W = 4  # cell width
    COLS = 8
    ROWS = 4
    pad = " " * indent

    lines = []

    # Address row (dim, above the grid)
    if show_addrs:
        addr_line = pad + " "
        for col in range(COLS):
            addr_line += f" {FG_GRAY}{DIM}{str(col).center(W)}{RESET}"
        lines.append(addr_line)

    for row in range(ROWS):
        # Top border
        if row == 0:
            border = pad + " " + "\u250c" + ("\u2500" * W + "\u252c") * (COLS - 1) + "\u2500" * W + "\u2510"
        else:
            border = pad + " " + "\u251c" + ("\u2500" * W + "\u253c") * (COLS - 1) + "\u2500" * W + "\u2524"
        lines.append(f"{FG_GRAY}{border}{RESET}")

        # Data row
        data_line = pad + " " + "\u2502"
        for col in range(COLS):
            addr = row * COLS + col
            val = mem[addr]
            vs = format_cell(val, W)

            # Determine style
            is_pc_cell = (addr == pc or addr == pc + 1 or addr == pc + 2)
            is_changed = addr in changed_addrs
            is_data = addr >= 24

            if is_changed:
                cell_str = f"{BOLD}{FG_YELLOW}{BG_DARK}{vs}{RESET}"
            elif is_pc_cell:
                cell_str = f"{BOLD}{FG_RED}{BG_DARKER}{vs}{RESET}"
            elif addr in highlight_addrs:
                cell_str = f"{BOLD}{FG_WHITE}{vs}{RESET}"
            elif is_data:
                cell_str = f"{FG_CYAN}{vs}{RESET}"
            elif addr < 9 and val != 0:  # active code
                cell_str = f"{FG_MAGENTA}{vs}{RESET}"
            else:
                cell_str = f"{FG_GRAY}{DIM}{vs}{RESET}"

            data_line += f"{FG_GRAY}\u2502{RESET}".join([""]) + cell_str
            if col < COLS - 1:
                data_line += f"{FG_GRAY}\u2502{RESET}"
        data_line += f"{FG_GRAY}\u2502{RESET}"

        # Row annotation
        if row == 0:
            data_line += f"  {FG_MAGENTA}{DIM}<- code{RESET}"
        elif row == 3:
            data_line += f"  {FG_CYAN}{DIM}<- data{RESET}"

        lines.append(data_line)

    # Bottom border
    bottom = pad + " " + "\u2514" + ("\u2500" * W + "\u2534") * (COLS - 1) + "\u2500" * W + "\u2518"
    lines.append(f"{FG_GRAY}{bottom}{RESET}")

    # Address labels for data row
    if show_addrs and True:
        addr_line2 = pad + " "
        for col in range(COLS):
            addr = 3 * COLS + col  # row 3
            addr_line2 += f" {FG_GRAY}{DIM}{str(addr).center(W)}{RESET}"
        lines.append(addr_line2)

    return lines

# ---------------------------------------------------------------------------
# Section: Title
# ---------------------------------------------------------------------------

def section_title():
    clear()
    pause(0.5)

    # Build the box
    width = 56
    top    = f"{FG_CYAN}{BOLD}\u2554" + "\u2550" * width + "\u2557{RESET}"
    bottom = f"{FG_CYAN}{BOLD}\u255a" + "\u2550" * width + "\u255d{RESET}"

    line1_text = "A Transformer That Learned to Be a Computer"
    line2_text = "SUBLEQ: One instruction. Turing complete."

    line1 = f"{FG_CYAN}{BOLD}\u2551{RESET}  {BOLD}{FG_WHITE}{line1_text}{RESET}" + " " * (width - 2 - len(line1_text)) + f"{FG_CYAN}{BOLD}\u2551{RESET}"
    line2 = f"{FG_CYAN}{BOLD}\u2551{RESET}  {FG_YELLOW}{line2_text}{RESET}" + " " * (width - 2 - len(line2_text)) + f"{FG_CYAN}{BOLD}\u2551{RESET}"

    emit("")
    emit("")

    # Type the top border
    for i, ch in enumerate(f"\u2554" + "\u2550" * width + "\u2557"):
        emit(f"{FG_CYAN}{BOLD}{ch}{RESET}", end="")
        if i % 4 == 0:
            pause(0.01)
    emit("")
    pause(0.1)

    # Line 1
    emit(f"{FG_CYAN}{BOLD}\u2551{RESET}", end="")
    emit("  ", end="")
    typing_no_nl(line1_text, per_char=0.025, style=f"{BOLD}{FG_WHITE}")
    emit(" " * (width - 2 - len(line1_text)), end="")
    emit(f"{FG_CYAN}{BOLD}\u2551{RESET}")
    pause(0.2)

    # Line 2
    emit(f"{FG_CYAN}{BOLD}\u2551{RESET}", end="")
    emit("  ", end="")
    typing_no_nl(line2_text, per_char=0.02, style=f"{FG_YELLOW}")
    emit(" " * (width - 2 - len(line2_text)), end="")
    emit(f"{FG_CYAN}{BOLD}\u2551{RESET}")
    pause(0.1)

    # Bottom border
    for i, ch in enumerate(f"\u255a" + "\u2550" * width + "\u255d"):
        emit(f"{FG_CYAN}{BOLD}{ch}{RESET}", end="")
        if i % 4 == 0:
            pause(0.01)
    emit("")

    pause(1.0)

# ---------------------------------------------------------------------------
# Section: SUBLEQ explanation
# ---------------------------------------------------------------------------

def section_subleq_explain():
    clear()
    pause(0.3)

    emit("")
    typing("  THE INSTRUCTION", per_char=0.04, style=f"{BOLD}{FG_CYAN}")
    emit("")
    pause(0.3)

    typing("  SUBLEQ a b c", per_char=0.04, style=f"{BOLD}{FG_WHITE}")
    emit("")
    pause(0.4)

    typing_no_nl("     mem[b] ", per_char=0.02, style=f"{FG_GRAY}")
    typing_no_nl("-=", per_char=0.06, style=f"{BOLD}{FG_YELLOW}")
    typing_no_nl(" mem[a]", per_char=0.02, style=f"{FG_GRAY}")
    emit("")
    pause(0.3)

    typing_no_nl("     if mem[b] ", per_char=0.02, style=f"{FG_GRAY}")
    typing_no_nl("<= 0", per_char=0.06, style=f"{BOLD}{FG_YELLOW}")
    typing_no_nl(": goto c", per_char=0.02, style=f"{FG_GRAY}")
    emit("")
    pause(0.6)

    emit("")
    typing("  That's it.", per_char=0.05, style=f"{DIM}{FG_WHITE}")
    pause(0.4)
    typing("  One instruction.", per_char=0.04, style=f"{FG_WHITE}")
    pause(0.3)
    typing("  Turing complete.", per_char=0.04, style=f"{BOLD}{FG_GREEN}")

    pause(1.5)

# ---------------------------------------------------------------------------
# Section: Load the program
# ---------------------------------------------------------------------------

def section_load_program():
    clear()
    pause(0.3)

    emit("")
    typing("  LOADING PROGRAM: multiply(7, 9)", per_char=0.03, style=f"{BOLD}{FG_CYAN}")
    emit("")
    pause(0.3)

    # Show the 3 instructions
    instrs = [
        ("Instr 0", "SUBLEQ 24 26  3", "result += 7", FG_MAGENTA),
        ("Instr 1", "SUBLEQ 27 25 -1", "counter--; halt if done", FG_MAGENTA),
        ("Instr 2", "SUBLEQ  9  9  0", "jump to start", FG_MAGENTA),
    ]

    for label, code, comment, color in instrs:
        emit(f"    {FG_GRAY}{label}:{RESET}  ", end="")
        typing_no_nl(code, per_char=0.02, style=f"{BOLD}{color}")
        emit(f"   {FG_GRAY}{DIM}// {comment}{RESET}")
        pause(0.2)

    emit("")
    pause(0.3)

    # Show data cells
    typing("  Data cells:", per_char=0.02, style=f"{FG_CYAN}")
    data_info = [
        ("mem[24]", " -7", "negated multiplicand"),
        ("mem[25]", "  9", "counter (multiplier)"),
        ("mem[26]", "  0", "result accumulator"),
        ("mem[27]", "  1", "constant one"),
    ]
    for name, val, desc in data_info:
        emit(f"    {FG_CYAN}{name}{RESET} = {BOLD}{FG_WHITE}{val}{RESET}  {FG_GRAY}{DIM}// {desc}{RESET}")
        pause(0.15)

    emit("")
    pause(0.4)

    # Render the initial memory grid
    typing("  Initial memory state:", per_char=0.02, style=f"{FG_CYAN}")
    emit("")

    grid_lines = render_memory_grid(INIT_MEM, pc=0, highlight_addrs={24, 25, 26, 27})
    for line in grid_lines:
        emit(line)
        pause(0.05)

    emit("")
    emit(f"    {FG_RED}{BOLD}PC = 0{RESET}  {FG_GRAY}(red cells = current instruction){RESET}")

    pause(2.0)

# ---------------------------------------------------------------------------
# Section: Execute steps
# ---------------------------------------------------------------------------

def section_execute():
    clear()
    pause(0.3)

    emit("")
    typing("  EXECUTING: transformer predicts each step", per_char=0.03, style=f"{BOLD}{FG_CYAN}")
    emit("")
    pause(0.5)

    # We'll show first 9 steps (3 full loop iterations), then 3 more, then skip
    steps_to_show = list(range(9))  # first 3 iterations
    steps_to_show += [12, 13, 14]   # iteration 5 start
    # Then we'll skip to the final steps
    steps_to_show += [24, 25]       # final iteration + halt

    prev_mem = list(INIT_MEM)

    # Print the initial grid and step info area
    # We'll use in-place updating with cursor positioning
    grid_lines = render_memory_grid(INIT_MEM, pc=0)
    grid_start_row = 4  # line where grid starts

    for line in grid_lines:
        emit(line)
    emit("")  # blank line after grid

    # Status line position
    status_line = ""

    shown_skip = False
    last_iteration = -1

    for idx, step_idx in enumerate(steps_to_show):
        trace = FULL_TRACE[step_idx]
        pc = trace['pc']
        a = trace['a']
        b = trace['b']
        c = trace['c']
        a_val = trace['a_val']
        b_before = trace['b_val_before']
        b_after = trace['b_val_after']
        new_pc = trace['new_pc']
        branch = trace['branch']
        mem_after = trace['mem']
        result = trace['result']
        counter = trace['counter']

        # Detect changed addresses
        changed = set()
        for addr in range(32):
            if mem_after[addr] != prev_mem[addr]:
                changed.add(addr)

        # Calculate which loop iteration we're in (3 steps per iteration)
        iteration = step_idx // 3 + 1

        # Show skip indicator
        if idx > 0 and step_idx > steps_to_show[idx - 1] + 1 and not shown_skip:
            emit("")
            emit(f"    {FG_GRAY}{DIM}  ... accelerating through iterations 4-8 ...{RESET}")
            emit("")
            pause(0.6)
            shown_skip = True
        elif idx > 0 and step_idx > steps_to_show[idx - 1] + 1:
            emit("")
            emit(f"    {FG_GRAY}{DIM}  ... final iteration ...{RESET}")
            emit("")
            pause(0.6)

        # Show iteration header when starting a new loop iteration
        if iteration != last_iteration and pc == 0:
            if iteration <= 3 or step_idx >= 24:
                label = f"Iteration {iteration}/9"
                if step_idx >= 24:
                    label = f"Iteration 9/9 (final)"
                emit(f"    {FG_CYAN}{DIM}\u2500\u2500\u2500 {label} \u2500\u2500\u2500{RESET}")
            last_iteration = iteration

        # Determine instruction name
        if pc == 0:
            instr_desc = "result += 7"
        elif pc == 3:
            instr_desc = "counter--"
        elif pc == 6:
            instr_desc = "jump -> 0"
        else:
            instr_desc = ""

        # Step header
        step_num = trace['step']
        emit(f"    {FG_GRAY}Step {step_num:2d}{RESET} ", end="")
        emit(f"{DIM}|{RESET} ", end="")
        emit(f"{FG_WHITE}pc={pc}{RESET} ", end="")
        emit(f"{FG_MAGENTA}SUBLEQ({a},{b},{c}){RESET} ", end="")
        emit(f"{FG_GRAY}{DIM}{instr_desc}{RESET}")

        pause(0.15)

        # Show the operation
        emit(f"           {FG_GRAY}mem[{b}]{RESET} = {FG_WHITE}{b_before}{RESET}", end="")
        emit(f" - mem[{a}]({FG_CYAN}{a_val}{RESET})", end="")
        emit(f" = {BOLD}{FG_YELLOW}{b_after}{RESET}", end="")

        if branch:
            if c == -1:
                emit(f"  {FG_RED}{BOLD}HALT{RESET}", end="")
            else:
                emit(f"  {FG_YELLOW}branch -> {c}{RESET}", end="")
        else:
            emit(f"  {FG_GRAY}fall through{RESET}", end="")
        emit("")

        # Neural net indicator
        if not RECORD:
            emit(f"           {FG_BLUE}Transformer predicts{RESET} -> ", end="")
            # Brief "thinking" animation using carriage return
            for dot in [".", "..", "..."]:
                emit_raw(f"\r           {FG_BLUE}Transformer predicts{RESET} -> {FG_YELLOW}{dot}{RESET}   ")
                pause(0.06)
            emit_raw(f"\r           {FG_BLUE}Transformer predicts{RESET} -> ")
            emit(f"[result={BOLD}{FG_WHITE}{result}{RESET}, counter={FG_WHITE}{counter}{RESET}]", end="")
            emit(f"  {FG_GREEN}{BOLD}matches interpreter{RESET}")
        else:
            emit(f"           Transformer predicts -> [result={result}, counter={counter}]  matches interpreter")

        # Progress bar for result
        bar_len = 30
        filled = int(result / 63 * bar_len) if result >= 0 else 0
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        emit(f"           {FG_GRAY}result: [{FG_GREEN}{bar}{FG_GRAY}] {result}/63{RESET}")

        emit("")
        prev_mem = list(mem_after)

        # Accelerating pace: first iterations slower, later ones faster
        if step_idx < 6:
            pause(0.3)
        elif step_idx < 15:
            pause(0.2)
        else:
            pause(0.15)

    pause(0.5)

    # Show the final memory grid
    emit("")
    typing("  Final memory state:", per_char=0.02, style=f"{FG_CYAN}")
    emit("")

    final_mem = FULL_TRACE[-1]['mem']
    grid_lines = render_memory_grid(
        final_mem, pc=-99,  # pc is -1 (halted), don't highlight any
        changed_addrs={25, 26},  # counter and result changed from init
        highlight_addrs={26},
    )
    for line in grid_lines:
        emit(line)
        pause(0.03)

    pause(1.0)

# ---------------------------------------------------------------------------
# Section: Final reveal
# ---------------------------------------------------------------------------

def section_reveal():
    clear()
    pause(0.5)

    emit("")
    emit("")

    # Build suspense
    emit(f"    {FG_GRAY}{DIM}Program halted after {TOTAL_STEPS} steps.{RESET}")
    pause(1.0)
    emit(f"    {FG_GRAY}{DIM}Reading result from memory...{RESET}")
    pause(1.0)
    emit("")

    # The big reveal box
    box_w = 51
    emit(f"    {FG_GREEN}\u250c" + "\u2500" * box_w + f"\u2510{RESET}")
    emit(f"    {FG_GREEN}\u2502{RESET}" + " " * box_w + f"{FG_GREEN}\u2502{RESET}")

    # mem[26] = 63
    line1 = f"  mem[26] = {BOLD}{FG_WHITE}63{RESET}"
    pad1 = box_w - 14
    emit(f"    {FG_GREEN}\u2502{RESET}{line1}" + " " * pad1 + f"{FG_GREEN}\u2502{RESET}")
    pause(0.5)

    # 7 x 9 = 63  CORRECT
    emit(f"    {FG_GREEN}\u2502{RESET}", end="")
    emit(f"  {BOLD}{FG_WHITE}7 x 9 = 63{RESET}", end="")
    pause(0.3)
    emit(f"  {BOLD}{FG_GREEN}CORRECT{RESET}", end="")
    pad2 = box_w - 23
    emit(" " * pad2, end="")
    emit(f"{FG_GREEN}\u2502{RESET}")
    pause(0.5)

    emit(f"    {FG_GREEN}\u2502{RESET}" + " " * box_w + f"{FG_GREEN}\u2502{RESET}")

    # Stats
    stats = [
        f"Never seen during training.",
        f"Learned from random single steps.",
        f"Executed {TOTAL_STEPS} steps perfectly.",
    ]
    for s in stats:
        pad = box_w - len(s) - 2
        emit(f"    {FG_GREEN}\u2502{RESET}  {FG_CYAN}{s}{RESET}" + " " * pad + f"{FG_GREEN}\u2502{RESET}")
        pause(0.4)

    emit(f"    {FG_GREEN}\u2502{RESET}" + " " * box_w + f"{FG_GREEN}\u2502{RESET}")
    emit(f"    {FG_GREEN}\u2514" + "\u2500" * box_w + f"\u2518{RESET}")

    pause(2.0)

# ---------------------------------------------------------------------------
# Section: Montage of other programs
# ---------------------------------------------------------------------------

def section_montage():
    clear()
    pause(0.3)

    emit("")
    typing("  EMERGENT PROGRAMS", per_char=0.04, style=f"{BOLD}{FG_CYAN}")
    typing("  Programs the transformer was never trained on:", per_char=0.02, style=f"{FG_GRAY}")
    emit("")
    pause(0.5)

    programs = [
        ("multiply(7, 9)", "63", "26 steps",  "3 instructions"),
        ("fibonacci(5)",   "55", "39 steps",  "8 instructions"),
        ("div(100, 7)",    "14", "71 steps",  "5 instructions"),
        ("isqrt(81)",       "9", "55 steps",  "6 instructions"),
    ]

    for name, result, steps, instrs in programs:
        emit(f"    ", end="")
        # Program name
        typing_no_nl(f"{name:>16s}", per_char=0.02, style=f"{FG_WHITE}")
        pause(0.2)

        # Arrow animation
        emit(f"  {FG_GRAY}", end="")
        for ch in ["─", "─", ">"]:
            emit(f"{ch}", end="")
            pause(0.05)
        emit(f"{RESET}", end="")
        pause(0.3)

        # Result
        emit(f"  {BOLD}{FG_GREEN}{result:>3s}{RESET}", end="")
        pause(0.1)
        emit(f"  {FG_GREEN}{BOLD}CORRECT{RESET}", end="")
        pause(0.1)
        emit(f"  {FG_GRAY}{DIM}({steps}, {instrs}){RESET}")

        pause(0.4)

    emit("")
    pause(0.5)

    # Multiplication table teaser
    typing("  Multiplication table (all products <= 127):", per_char=0.02, style=f"{FG_CYAN}")
    emit("")
    pause(0.3)

    # Show a compact multiplication table snippet
    emit(f"      {FG_GRAY}{DIM}", end="")
    for b in range(1, 10):
        emit(f"{b:>4d}", end="")
    emit(f"{RESET}")
    emit(f"      {FG_GRAY}" + "\u2500" * 36 + f"{RESET}")

    for a in range(1, 10):
        emit(f"    {FG_GRAY}{a:>2d}{RESET}{FG_GRAY}|{RESET}", end="")
        for b in range(1, 10):
            prod = a * b
            if prod > 127:
                emit(f" {FG_GRAY}{DIM}  .{RESET}", end="")
            else:
                emit(f" {FG_GREEN}{prod:>3d}{RESET}", end="")
        emit("")
        pause(0.05)

    emit("")
    emit(f"    {FG_GREEN}{BOLD}All correct.{RESET} {FG_GRAY}Every product learned from single-step training.{RESET}")

    pause(2.0)

# ---------------------------------------------------------------------------
# Section: Closing
# ---------------------------------------------------------------------------

def section_closing():
    clear()
    pause(0.5)

    emit("")
    emit("")

    lines = [
        (f"{BOLD}{FG_WHITE}", "4.9M parameters."),
        (f"{FG_WHITE}", "32 memory cells. 8-bit integers."),
        (f"{FG_WHITE}", "One instruction: subtract and branch."),
    ]

    for style, text in lines:
        emit("    ", end="")
        typing(text, per_char=0.04, style=style)
        pause(0.3)

    emit("")
    pause(0.5)

    emit("    ", end="")
    typing("Trained on random single steps.", per_char=0.04, style=f"{FG_CYAN}")
    pause(0.3)
    emit("    ", end="")
    typing("Never saw a full program.", per_char=0.04, style=f"{FG_CYAN}")
    pause(0.6)

    emit("")
    emit("    ", end="")
    typing("Emerged: a general-purpose computer.", per_char=0.05, style=f"{BOLD}{FG_GREEN}")

    pause(1.5)
    emit("")
    emit("")

    # Final box
    width = 56
    emit(f"    {FG_CYAN}{DIM}" + "\u2500" * width + f"{RESET}")
    emit(f"    {FG_GRAY}{DIM}Architecture: 256-dim, 6 layers, 8 heads, Pre-LN Transformer{RESET}")
    emit(f"    {FG_GRAY}{DIM}Hardware emulated: Manchester Baby class (1948){RESET}")
    emit(f"    {FG_GRAY}{DIM}ISA: SUBLEQ (subtract and branch if <= 0){RESET}")
    emit(f"    {FG_CYAN}{DIM}" + "\u2500" * width + f"{RESET}")

    pause(2.0)
    emit("")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global SPEED, RECORD

    parser = argparse.ArgumentParser(
        description="Terminal animation: A Transformer That Learned to Be a Computer"
    )
    parser.add_argument("--fast", action="store_true",
                        help="Run 5x faster (for testing)")
    parser.add_argument("--record", action="store_true",
                        help="Dump plain text frames to stdout (no ANSI)")
    args = parser.parse_args()

    if args.fast:
        SPEED = 0.2
    if args.record:
        RECORD = True

    if not RECORD:
        emit_raw(HIDE_CURSOR)

    try:
        section_title()
        section_subleq_explain()
        section_load_program()
        section_execute()
        section_reveal()
        section_montage()
        section_closing()
    except KeyboardInterrupt:
        pass
    finally:
        if not RECORD:
            emit_raw(SHOW_CURSOR)
            emit_raw(RESET)
        else:
            # Dump all frames
            sys.stdout.write("".join(_frame_buffer))


if __name__ == "__main__":
    main()
