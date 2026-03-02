#!/usr/bin/env python3
"""
Interactive SUBLEQ computer — watch a neural network execute programs.

A terminal REPL that lets you load built-in programs or enter custom memory,
then step through execution watching the transformer predict each state
transition, compared side-by-side against the ground-truth interpreter.

Usage:
    python play.py                              # default checkpoint
    python play.py checkpoints/best_model.pt    # specific checkpoint

Commands inside the REPL:
    multiply 7 9     Run multiplication program
    fibonacci 3      Run Fibonacci program
    divide 100 7     Run integer division
    isqrt 81         Run integer square root
    negate -42       Run negation
    add 30 50        Run addition
    countdown 10     Run countdown
    random           Run a random program
    custom           Enter custom memory values
    step             Execute one step (default mode)
    run              Run to halt
    run slow         Run with 0.3s delay per step
    reset            Reset current program
    help             Show this help
    quit             Exit
"""

import sys
import os
import time
import random
import argparse

import torch

from subleq import (
    MiniSUBLEQTransformer, step as subleq_step, run as subleq_run,
    MEM_SIZE, VALUE_MIN, VALUE_MAX, CODE_SIZE, DATA_START,
    encode, decode, SEQ_LEN, VOCAB_SIZE,
    make_negate, make_addition, make_countdown, make_multiply,
    make_fibonacci, make_div, make_isqrt, make_halt,
    generate_random_program,
)

# ── ANSI escape codes ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAGENTA = "\033[95m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"
BG_YELLOW = "\033[43m\033[30m"  # yellow background, black text


def auto_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model = MiniSUBLEQTransformer(
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model


def model_step(model, memory, pc, device='cpu'):
    inp = encode(memory, pc).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
    pred_tokens = logits.argmax(dim=-1).squeeze(0)
    return decode(pred_tokens)


# ── Display helpers ─────────────────────────────────────────────────

def fmt_cell(val, cell_idx, pc, changed_cells=None):
    """Format a single memory cell with color coding."""
    s = f"{val:4d}"

    # Highlight changed cells
    if changed_cells and cell_idx in changed_cells:
        return f"{BG_YELLOW}{s}{RESET}"

    # Color by region
    if cell_idx < CODE_SIZE:
        # Code region — highlight current instruction
        if pc >= 0 and pc <= cell_idx < pc + 3:
            return f"{CYAN}{BOLD}{s}{RESET}"
        return f"{DIM}{s}{RESET}"
    else:
        # Data region
        return f"{GREEN}{s}{RESET}"


def show_memory_grid(mem, pc, changed_cells=None, label="Memory"):
    """Display memory as a 4x8 grid with color coding."""
    print(f"\n  {BOLD}{label}{RESET}  (pc={pc})")
    print(f"  {'':4s}", end="")
    for col in range(8):
        print(f"  {DIM}[{col:2d}]{RESET}", end="")
    print()

    for row in range(4):
        row_start = row * 8
        region = f"{DIM}code{RESET}" if row_start < CODE_SIZE else f"{GREEN}data{RESET}"
        print(f"  {row_start:2d} {region[:20]:>4s}", end="")
        for col in range(8):
            idx = row_start + col
            print(f"  {fmt_cell(mem[idx], idx, pc, changed_cells)}", end="")
        print()


def show_instruction(mem, pc):
    """Display the current SUBLEQ instruction being executed."""
    if pc < 0 or pc + 2 >= MEM_SIZE:
        print(f"\n  {RED}HALTED{RESET} (pc={pc} out of bounds)")
        return False

    a, b, c = mem[pc], mem[pc + 1], mem[pc + 2]

    if a < 0 or a >= MEM_SIZE or b < 0 or b >= MEM_SIZE:
        print(f"\n  {RED}HALTED{RESET} (invalid addresses a={a}, b={b})")
        return False

    result = mem[b] - mem[a]
    branch = "yes" if result <= 0 else "no"
    target = c if result <= 0 else pc + 3

    print(f"\n  {CYAN}Instruction at pc={pc}:{RESET}")
    print(f"    SUBLEQ  a={a}  b={b}  c={c}")
    print(f"    mem[{b}] = mem[{b}]({mem[b]:d}) - mem[{a}]({mem[a]:d}) = {result:d}")
    print(f"    result <= 0? {branch} -> ", end="")
    if result <= 0:
        if c < 0:
            print(f"{RED}HALT{RESET} (c={c})")
        else:
            print(f"jump to pc={c}")
    else:
        print(f"next (pc={pc+3})")

    return True


def show_step_result(model_mem, model_pc, true_mem, true_pc, step_num):
    """Compare model prediction vs ground truth."""
    match = (model_mem == true_mem and model_pc == true_pc)

    changed = set()
    for i in range(MEM_SIZE):
        if model_mem[i] != true_mem[i]:
            changed.add(i)

    if match:
        print(f"\n  {GREEN}Step {step_num}: Model matches interpreter perfectly{RESET}")
    else:
        print(f"\n  {RED}Step {step_num}: MISMATCH!{RESET}")
        if model_pc != true_pc:
            print(f"    PC: model={model_pc}, truth={true_pc}")
        for i in changed:
            region = "code" if i < CODE_SIZE else "data"
            print(f"    mem[{i}] ({region}): model={model_mem[i]}, truth={true_mem[i]}")

    return match


# ── Program loaders ────────────────────────────────────────────────

EXAMPLES = {
    'multiply':  "Multiplication via repeated addition",
    'fibonacci': "Fibonacci sequence (alternating accumulation)",
    'divide':    "Integer division via repeated subtraction",
    'isqrt':     "Integer square root via odd number sum",
    'negate':    "Negate a value",
    'add':       "Addition of two values",
    'countdown': "Count down to zero",
    'random':    "Random SUBLEQ program",
}


def load_program(cmd, args_list):
    """Parse a command and return (mem, pc, description, result_info)."""
    try:
        if cmd == 'multiply':
            a = int(args_list[0]) if args_list else 7
            b = int(args_list[1]) if len(args_list) > 1 else 9
            assert abs(a) * abs(b) <= VALUE_MAX, f"Overflow: {a}*{b} > {VALUE_MAX}"
            mem, pc, r = make_multiply(abs(a), abs(b))
            return mem, pc, f"multiply({a}, {b})", f"result in mem[{r}]"

        elif cmd == 'fibonacci':
            n = int(args_list[0]) if args_list else 3
            mem, pc, ra, rb = make_fibonacci(n)
            return mem, pc, f"fibonacci(n={n})", f"F({2*n}) in mem[{ra}], F({2*n+1}) in mem[{rb}]"

        elif cmd == 'divide':
            a = int(args_list[0]) if args_list else 100
            b = int(args_list[1]) if len(args_list) > 1 else 7
            mem, pc, r = make_div(a, b)
            return mem, pc, f"divide({a}, {b})", f"quotient in mem[{r}]"

        elif cmd == 'isqrt':
            n = int(args_list[0]) if args_list else 81
            mem, pc, r = make_isqrt(n)
            return mem, pc, f"isqrt({n})", f"result in mem[{r}]"

        elif cmd == 'negate':
            v = int(args_list[0]) if args_list else -42
            mem, pc, r = make_negate(v)
            return mem, pc, f"negate({v})", f"result in mem[{r}]"

        elif cmd == 'add':
            a = int(args_list[0]) if args_list else 30
            b = int(args_list[1]) if len(args_list) > 1 else 50
            mem, pc, r = make_addition(a, b)
            return mem, pc, f"add({a}, {b})", f"result in mem[{r}]"

        elif cmd == 'countdown':
            n = int(args_list[0]) if args_list else 10
            mem, pc, r = make_countdown(n)
            return mem, pc, f"countdown({n})", f"counter in mem[{r}]"

        elif cmd == 'random':
            n_instr = int(args_list[0]) if args_list else random.randint(2, 6)
            mem, pc = generate_random_program(n_instr)
            return mem, pc, f"random program ({n_instr} instructions)", "watch it run"

        else:
            return None, None, None, None

    except (ValueError, AssertionError) as e:
        print(f"  {RED}Error: {e}{RESET}")
        return None, None, None, None


def custom_program():
    """Let the user enter custom memory values."""
    print(f"\n  {CYAN}Enter 32 memory values (space-separated, range [{VALUE_MIN}, {VALUE_MAX}]):{RESET}")
    print(f"  {DIM}(Enter fewer values and the rest will be zero){RESET}")
    try:
        line = input(f"  {BOLD}mem>{RESET} ").strip()
        if not line:
            return None, None
        vals = [int(x) for x in line.split()]
        mem = [0] * MEM_SIZE
        for i, v in enumerate(vals[:MEM_SIZE]):
            mem[i] = max(VALUE_MIN, min(VALUE_MAX, v))

        pc_str = input(f"  {BOLD}pc>{RESET} ").strip()
        pc = int(pc_str) if pc_str else 0
        return mem, pc
    except (ValueError, EOFError):
        return None, None


# ── Main REPL ──────────────────────────────────────────────────────

def show_help():
    print(f"\n  {CYAN}{BOLD}Available programs:{RESET}")
    for name, desc in EXAMPLES.items():
        print(f"    {BOLD}{name:12s}{RESET}  {desc}")
    print(f"    {BOLD}{'custom':12s}{RESET}  Enter your own memory values")
    print(f"\n  {CYAN}{BOLD}Controls:{RESET}")
    print(f"    {BOLD}step{RESET}          Execute one SUBLEQ step  (or just press Enter)")
    print(f"    {BOLD}run{RESET}           Run to halt")
    print(f"    {BOLD}run slow{RESET}      Run with 0.3s delay per step")
    print(f"    {BOLD}reset{RESET}         Restart current program")
    print(f"    {BOLD}help{RESET}          Show this help")
    print(f"    {BOLD}quit{RESET}          Exit")


def banner():
    print(f"""
{CYAN}{BOLD}  ╔═══════════════════════════════════════════════════════╗
  ║          SUBLEQ  NEURAL  COMPUTER                   ║
  ║  A transformer that learned to be a general-purpose  ║
  ║  computer from data alone. Type 'help' for commands. ║
  ╚═══════════════════════════════════════════════════════╝{RESET}
""")


def repl(model, device):
    banner()
    show_help()

    mem = None
    pc = 0
    orig_mem = None
    orig_pc = 0
    step_count = 0
    desc = ""
    result_info = ""

    while True:
        try:
            if mem is None:
                prompt = f"\n  {BOLD}subleq>{RESET} "
            else:
                prompt = f"\n  {BOLD}[step {step_count}]>{RESET} "
            line = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {DIM}Goodbye!{RESET}")
            break

        if not line:
            if mem is not None:
                line = "step"
            else:
                continue

        parts = line.split()
        cmd = parts[0]
        args_list = parts[1:]

        # ── Navigation commands ──
        if cmd in ('quit', 'exit', 'q'):
            print(f"\n  {DIM}Goodbye!{RESET}")
            break

        elif cmd == 'help':
            show_help()

        elif cmd == 'reset':
            if orig_mem is not None:
                mem = list(orig_mem)
                pc = orig_pc
                step_count = 0
                print(f"\n  {YELLOW}Reset: {desc}{RESET}")
                show_memory_grid(mem, pc, label=desc)
            else:
                print(f"  {DIM}No program loaded. Try: multiply 7 9{RESET}")

        # ── Load programs ──
        elif cmd == 'custom':
            m, p = custom_program()
            if m is not None:
                mem, pc = m, p
                orig_mem, orig_pc = list(m), p
                step_count = 0
                desc = "custom program"
                result_info = ""
                show_memory_grid(mem, pc, label=desc)

        elif cmd in EXAMPLES:
            m, p, d, r = load_program(cmd, args_list)
            if m is not None:
                mem, pc = m, p
                orig_mem, orig_pc = list(m), p
                step_count = 0
                desc = d
                result_info = r
                print(f"\n  {CYAN}{BOLD}Loaded: {desc}{RESET}")
                if result_info:
                    print(f"  {DIM}{result_info}{RESET}")
                show_memory_grid(mem, pc, label=desc)

        # ── Execution commands ──
        elif cmd == 'step':
            if mem is None:
                print(f"  {DIM}No program loaded. Try: multiply 7 9{RESET}")
                continue

            if not show_instruction(mem, pc):
                continue

            # Run model
            model_mem, model_pc = model_step(model, mem, pc, device)

            # Run interpreter
            true_mem, true_pc, halted = subleq_step(mem, pc)

            step_count += 1
            match = show_step_result(model_mem, model_pc, true_mem, true_pc, step_count)

            # Compute changed cells for highlighting
            changed = set()
            for i in range(MEM_SIZE):
                if mem[i] != model_mem[i]:
                    changed.add(i)

            # Advance state using model's prediction
            mem = model_mem
            pc = model_pc

            show_memory_grid(mem, pc, changed_cells=changed, label=f"{desc} (step {step_count})")

            if halted or pc < 0 or pc + 2 >= MEM_SIZE:
                print(f"\n  {YELLOW}{BOLD}Program halted after {step_count} steps.{RESET}")
                if result_info:
                    print(f"  {result_info}")

        elif cmd == 'run':
            if mem is None:
                print(f"  {DIM}No program loaded. Try: multiply 7 9{RESET}")
                continue

            slow = len(args_list) > 0 and args_list[0] == 'slow'
            delay = 0.3 if slow else 0.0
            max_run = 500
            matches = 0
            mismatches = 0

            for _ in range(max_run):
                if pc < 0 or pc + 2 >= MEM_SIZE:
                    break
                a_val, b_val = mem[pc], mem[pc + 1]
                if a_val < 0 or a_val >= MEM_SIZE or b_val < 0 or b_val >= MEM_SIZE:
                    break

                model_mem, model_pc = model_step(model, mem, pc, device)
                true_mem, true_pc, halted = subleq_step(mem, pc)
                step_count += 1

                match = (model_mem == true_mem and model_pc == true_pc)
                if match:
                    matches += 1
                else:
                    mismatches += 1

                if slow:
                    status = f"{GREEN}OK{RESET}" if match else f"{RED}MISMATCH{RESET}"
                    print(f"  step {step_count:3d}: pc={pc:2d} -> {model_pc:2d}  "
                          f"mem[{b_val}]: {mem[b_val]:4d} -> {model_mem[b_val]:4d}  {status}")
                    time.sleep(delay)

                mem = model_mem
                pc = model_pc

                if halted:
                    break

            # Show final state
            print(f"\n  {YELLOW}{BOLD}Program halted after {step_count} total steps.{RESET}")
            print(f"  Accuracy: {GREEN}{matches}{RESET} correct, "
                  f"{RED if mismatches > 0 else DIM}{mismatches}{RESET} mismatches "
                  f"out of {matches + mismatches} steps")

            show_memory_grid(mem, pc, label=f"{desc} (final)")
            if result_info:
                print(f"  {result_info}")

        else:
            # Maybe it's a program command with no space
            m, p, d, r = load_program(cmd, args_list)
            if m is not None:
                mem, pc = m, p
                orig_mem, orig_pc = list(m), p
                step_count = 0
                desc = d
                result_info = r
                print(f"\n  {CYAN}{BOLD}Loaded: {desc}{RESET}")
                if result_info:
                    print(f"  {DIM}{result_info}{RESET}")
                show_memory_grid(mem, pc, label=desc)
            else:
                print(f"  {DIM}Unknown command: {cmd}. Type 'help' for options.{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive SUBLEQ neural computer")
    parser.add_argument("model_path", nargs='?', default="checkpoints/best_model.pt",
                        help="Path to model checkpoint (default: checkpoints/best_model.pt)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = auto_device()

    if not os.path.exists(args.model_path):
        print(f"{RED}Error: checkpoint not found at {args.model_path}{RESET}")
        print("Run 'make train' first, or specify a path: python play.py <path>")
        sys.exit(1)

    print(f"  {DIM}Loading model from {args.model_path}...{RESET}")
    model = load_model(args.model_path, args.device)
    print(f"  {GREEN}Model loaded ({model.count_params():,} parameters, device={args.device}){RESET}")

    repl(model, args.device)
