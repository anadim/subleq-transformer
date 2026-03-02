"""
Program generators for byte-tokenized SUBLEQ (32 cells, 8-bit values).

Memory layout:
    Cells 0-23: Code (up to 8 three-word instructions)
    Cells 24-31: Data (8 cells)
"""

import random
from .interpreter import MEM_SIZE, DATA_START, VALUE_MIN, VALUE_MAX, clamp


def _pad(mem):
    while len(mem) < MEM_SIZE:
        mem.append(0)
    return mem[:MEM_SIZE]


def make_negate(val):
    """Negate: result = -val. Cell 24=input, cell 25=result."""
    val = clamp(val)
    mem = [0] * MEM_SIZE
    mem[0] = 25; mem[1] = 25; mem[2] = 3     # clear result
    mem[3] = 24; mem[4] = 25; mem[5] = 6     # result -= input = -input
    mem[6] = 26; mem[7] = 26; mem[8] = -1    # halt
    mem[24] = val
    mem[25] = 0
    return mem, 0, 25


def make_addition(a, b):
    """Add: result = a + b. Cell 24=a, 25=b, 26=result, 27=temp."""
    a, b = clamp(a), clamp(b)
    mem = [0] * MEM_SIZE
    mem[0] = 27; mem[1] = 27; mem[2] = 3     # clear temp
    mem[3] = 24; mem[4] = 27; mem[5] = 6     # temp = -a
    mem[6] = 27; mem[7] = 26; mem[8] = 9     # result -= (-a) = b + a
    mem[9] = 26; mem[10] = 26; mem[11] = -1  # halt
    mem[24] = a
    mem[25] = b
    mem[26] = b   # result starts as b
    mem[27] = 0
    return mem, 0, 26


def make_countdown(start):
    """Count down from start to 0. Cell 24=counter, 25=one."""
    start = max(1, min(start, VALUE_MAX))
    mem = [0] * MEM_SIZE
    mem[0] = 25; mem[1] = 24; mem[2] = -1    # counter -= 1, halt if <=0
    mem[3] = 26; mem[4] = 26; mem[5] = 0     # jump back to 0
    mem[24] = start
    mem[25] = 1
    mem[26] = 0
    return mem, 0, 24


def make_multiply(a, b):
    """Multiply: result = a * b (repeated addition).
    Cell 24=-a, 25=counter(b), 26=result, 27=const 1.
    3 instructions, 3*b steps.
    """
    a, b = abs(clamp(a)), abs(clamp(b))
    assert a * b <= VALUE_MAX, f"Overflow: {a}*{b}={a*b}"
    mem = [0] * MEM_SIZE
    mem[0] = 24; mem[1] = 26; mem[2] = 3     # result -= (-a) = result + a
    mem[3] = 27; mem[4] = 25; mem[5] = -1    # counter -= 1, halt if <=0
    mem[6] = 9;  mem[7] = 9;  mem[8] = 0     # unconditional jump to 0
    mem[24] = clamp(-a)
    mem[25] = b
    mem[26] = 0
    mem[27] = 1
    return mem, 0, 26


def make_fibonacci(n):
    """Compute Fibonacci numbers using alternating a+=b, b+=a.

    Each loop iteration advances two Fibonacci steps:
      a += b  (a = F_{2k})
      b += a  (b = F_{2k+1})

    After n iterations: a = F(2n), b = F(2n+1).
    Data: 27=a, 28=b, 29=counter, 30=one, 31=temp.
    """
    n = max(1, min(n, 10))
    mem = [0] * MEM_SIZE

    # --- First half: a += b ---
    # Instr 0 (pc=0): clear temp
    mem[0] = 31; mem[1] = 31; mem[2] = 3
    # Instr 1 (pc=3): temp -= b -> temp = -b
    mem[3] = 28; mem[4] = 31; mem[5] = 6
    # Instr 2 (pc=6): a -= temp -> a += b
    mem[6] = 31; mem[7] = 27; mem[8] = 9

    # --- Second half: b += a ---
    # Instr 3 (pc=9): clear temp
    mem[9] = 31; mem[10] = 31; mem[11] = 12
    # Instr 4 (pc=12): temp -= a -> temp = -a (which is now -new_a)
    mem[12] = 27; mem[13] = 31; mem[14] = 15
    # Instr 5 (pc=15): b -= temp -> b += a
    mem[15] = 31; mem[16] = 28; mem[17] = 18

    # --- Counter and loop ---
    # Instr 6 (pc=18): counter -= one, halt if <=0
    mem[18] = 30; mem[19] = 29; mem[20] = -1
    # If counter > 0: pc = 21

    # Instr 7 (pc=21): trampoline - jump back to 0
    mem[21] = 31; mem[22] = 31; mem[23] = 0
    # temp -= temp -> 0, always branch to 0

    # Data
    mem[27] = 0   # a = F(0)
    mem[28] = 1   # b = F(1)
    mem[29] = n   # counter (number of double-steps)
    mem[30] = 1   # constant one
    mem[31] = 0   # temp

    return mem, 0, 27, 28  # mem, pc, result_addr_a, result_addr_b


def make_div(a, b):
    """Integer division a // b via repeated subtraction.
    Uses n+1 trick to handle SUBLEQ's <= 0 branch correctly.
    """
    a, b = abs(a), abs(b)
    assert b > 0 and a + 1 <= VALUE_MAX
    mem = [0] * MEM_SIZE
    # Instr 0: n -= b, halt if <= 0
    mem[0] = 25; mem[1] = 24; mem[2] = -1
    # Instr 1: clear temp
    mem[3] = 29; mem[4] = 29; mem[5] = 6
    # Instr 2: temp = -one
    mem[6] = 27; mem[7] = 29; mem[8] = 9
    # Instr 3: quotient -= temp (quotient += 1)
    mem[9] = 29; mem[10] = 26; mem[11] = 12
    # Instr 4: trampoline to 0
    mem[12] = 29; mem[13] = 29; mem[14] = 0

    mem[24] = a + 1  # +1 to fix SUBLEQ's <= 0 branch
    mem[25] = b
    mem[26] = 0      # quotient
    mem[27] = 1      # constant 1
    mem[29] = 0      # temp
    return mem, 0, 26


def make_isqrt(n):
    """Integer square root via 1+3+5+7+... (sum of odd numbers = k^2).
    Uses n+1 trick for correct results on perfect squares.
    """
    n = max(0, min(n, VALUE_MAX - 1))  # -1 because we add 1
    mem = [0] * MEM_SIZE
    # Instr 0: n -= odd, halt if <= 0
    mem[0] = 25; mem[1] = 24; mem[2] = -1
    # Instr 1: clear temp
    mem[3] = 29; mem[4] = 29; mem[5] = 6
    # Instr 2: temp = -one
    mem[6] = 27; mem[7] = 29; mem[8] = 9
    # Instr 3: count -= temp (count += 1)
    mem[9] = 29; mem[10] = 26; mem[11] = 12
    # Instr 4: odd -= neg2 (odd += 2)
    mem[12] = 28; mem[13] = 25; mem[14] = 15
    # Instr 5: trampoline to 0
    mem[15] = 29; mem[16] = 29; mem[17] = 0

    mem[24] = n + 1  # +1 trick
    mem[25] = 1      # current odd number
    mem[26] = 0      # count (result)
    mem[27] = 1      # constant 1
    mem[28] = -2     # constant -2 (for odd += 2)
    mem[29] = 0      # temp
    return mem, 0, 26


def make_halt():
    """Immediate halt."""
    mem = [0] * MEM_SIZE
    mem[0] = 24; mem[1] = 24; mem[2] = -1
    return mem, 0


def generate_random_program(num_instructions=None, value_range=30):
    """Generate a random SUBLEQ program for 32 cells."""
    if num_instructions is None:
        num_instructions = random.randint(1, 8)
    num_instructions = min(num_instructions, 8)

    mem = [0] * MEM_SIZE

    for i in range(num_instructions):
        base = i * 3
        if random.random() < 0.7:
            a = random.randint(DATA_START, MEM_SIZE - 1)
        else:
            a = random.randint(0, MEM_SIZE - 1)
        if random.random() < 0.7:
            b = random.randint(DATA_START, MEM_SIZE - 1)
        else:
            b = random.randint(0, MEM_SIZE - 1)
        if random.random() < 0.15:
            c = -1
        else:
            c = random.randint(0, num_instructions - 1) * 3
        mem[base] = a
        mem[base + 1] = b
        mem[base + 2] = c

    for j in range(DATA_START, MEM_SIZE):
        mem[j] = random.randint(-value_range, value_range)

    pc = random.randint(0, num_instructions - 1) * 3
    return mem, pc


def generate_random_state(num_instructions=None, value_range=30):
    """Generate a random valid state that won't immediately halt."""
    for _ in range(100):
        mem, pc = generate_random_program(num_instructions, value_range)
        if 0 <= pc and pc + 2 < MEM_SIZE:
            a, b = mem[pc], mem[pc + 1]
            if 0 <= a < MEM_SIZE and 0 <= b < MEM_SIZE:
                return mem, pc
    return make_halt()
