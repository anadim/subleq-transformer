"""
SUBLEQ program generators for the 416-cell machine.

Memory layout: cells 0-383 = code (128 three-word instructions),
               cells 384-415 = data (32 cells).
"""

import random
try:
    from .interpreter import MEM_SIZE, CODE_SIZE, DATA_START, VALUE_MIN, VALUE_MAX, clamp
except ImportError:
    from interpreter import MEM_SIZE, CODE_SIZE, DATA_START, VALUE_MIN, VALUE_MAX, clamp


def _fresh_mem():
    """Return zeroed 416-cell memory."""
    return [0] * MEM_SIZE


def make_negate(v):
    """Negate a value: result = -v.

    3-instruction program (9 code words):
      0: SUBLEQ r, r, 3    (clear r; r=0 <= 0, goto 3)
      3: SUBLEQ v, r, 6    (r -= v = -v; -v <= 0 if v >= 0, goto 6)
      6: SUBLEQ z, z, -1   (halt: jump to -1)

    Data: cell 384 = v, cell 385 = result, cell 386 = 0 (for halt).
    """
    mem = _fresh_mem()
    r = DATA_START + 1   # 385
    v_addr = DATA_START  # 384
    z = DATA_START + 2   # 386

    # Instruction 0: SUBLEQ r r 3
    mem[0], mem[1], mem[2] = r, r, 3
    # Instruction 1: SUBLEQ v_addr r 6
    mem[3], mem[4], mem[5] = v_addr, r, 6
    # Instruction 2: SUBLEQ z z -1 (halt)
    mem[6], mem[7], mem[8] = z, z, -1

    mem[v_addr] = clamp(v)
    mem[r] = 0

    return mem, 0, r  # memory, pc, result_addr


def make_addition(a, b):
    """Add two values: result = a + b.

    Uses negate-then-subtract: result = 0 - (-a) - (-b) = a + b.
    Data: 384=a, 385=b, 386=result, 387=zero.
    """
    mem = _fresh_mem()
    a_addr = DATA_START       # 384
    b_addr = DATA_START + 1   # 385
    r_addr = DATA_START + 2   # 386
    z_addr = DATA_START + 3   # 387

    # Clear result
    # 0: SUBLEQ r r 3
    mem[0], mem[1], mem[2] = r_addr, r_addr, 3
    # Subtract a from result: result -= a => result = -a
    # 3: SUBLEQ a r 6
    mem[3], mem[4], mem[5] = a_addr, r_addr, 6
    # Subtract b from result: result -= b => result = -a - b
    # 6: SUBLEQ b r 9
    mem[6], mem[7], mem[8] = b_addr, r_addr, 9
    # Negate result: need temp = 0, temp -= result, result = 0, result -= temp
    t_addr = DATA_START + 4  # 388
    # 9: SUBLEQ t t 12  (clear temp)
    mem[9], mem[10], mem[11] = t_addr, t_addr, 12
    # 12: SUBLEQ r t 15  (temp = -result = a+b)
    mem[12], mem[13], mem[14] = r_addr, t_addr, 15
    # 15: SUBLEQ r r 18  (clear result)
    mem[15], mem[16], mem[17] = r_addr, r_addr, 18
    # 18: SUBLEQ t r 21  (result = -temp = -(-(a+b)) = a+b)
    mem[18], mem[19], mem[20] = t_addr, r_addr, 21
    # 21: SUBLEQ z z -1  (halt)
    mem[21], mem[22], mem[23] = z_addr, z_addr, -1

    mem[a_addr] = clamp(a)
    mem[b_addr] = clamp(b)
    mem[r_addr] = 0
    mem[z_addr] = 0

    return mem, 0, r_addr


def make_copy_countdown(n):
    """Copy a value and count down from n to 0.

    Exercises loop behavior. Result cell ends at 0.
    Data: 384=counter, 385=one, 386=zero.
    """
    mem = _fresh_mem()
    ctr = DATA_START       # 384 (counter)
    one = DATA_START + 1   # 385 (constant 1)
    z = DATA_START + 2     # 386

    mem[0], mem[1], mem[2] = one, ctr, 6
    mem[3], mem[4], mem[5] = z, z, 0
    mem[6], mem[7], mem[8] = z, z, -1

    mem[ctr] = clamp(n)
    mem[one] = 1
    mem[z] = 0

    return mem, 0, ctr


def make_multiply(a, b):
    """Multiply: result = a * b via repeated addition.

    Uses a loop that adds a to result, b times.
    Data: 384=a, 385=b(counter), 386=result, 387=one, 388=temp, 389=zero.
    """
    mem = _fresh_mem()
    a_addr = DATA_START       # 384
    b_addr = DATA_START + 1   # 385 (counter, decremented)
    r_addr = DATA_START + 2   # 386 (result, accumulates)
    one    = DATA_START + 3   # 387
    t_addr = DATA_START + 4   # 388 (temp for negation)
    z_addr = DATA_START + 5   # 389

    pc = 0
    def emit(a_, b_, c_):
        nonlocal pc
        mem[pc], mem[pc+1], mem[pc+2] = a_, b_, c_
        pc += 3

    # Clear temp
    emit(t_addr, t_addr, pc + 3)          # 0: t = 0
    # temp -= a => temp = -a
    emit(a_addr, t_addr, pc + 3)          # 3: t -= a
    # result -= temp => result += a
    emit(t_addr, r_addr, pc + 3)          # 6: r -= t (r += a)
    # decrement b
    emit(one, b_addr, pc + 6)             # 9: b -= 1; if b<=0, goto pc+6 (halt)
    # loop
    emit(z_addr, z_addr, 0)               # 12: jump to 0
    # halt
    emit(z_addr, z_addr, -1)              # 15: halt

    mem[a_addr] = clamp(a)
    mem[b_addr] = clamp(b)
    mem[r_addr] = 0
    mem[one] = 1
    mem[z_addr] = 0

    return mem, 0, r_addr


def make_bubble_sort(values):
    """Bubble sort via SUBLEQ using self-modifying code.

    Returns (memory, pc, data_start_addr, n_elements).
    After running, memory[data_start:data_start+n] is sorted ascending.
    """
    n = len(values)
    if n <= 1:
        mem = _fresh_mem()
        if n == 1:
            mem[DATA_START] = clamp(values[0])
        return mem, -1, DATA_START, n

    mem = _fresh_mem()
    arr = DATA_START
    for i, v in enumerate(values):
        mem[arr + i] = clamp(v)

    # Use a SUBLEQ "assembler" approach: build the program as a list
    # of (a, b, c) triples, then lay them into memory.
    code = []  # list of [a, b, c] triples; c can be a label string
    labels = {}  # label -> instruction index
    fixups = []  # (instr_idx, field_idx, label) for forward references

    def label(name):
        labels[name] = len(code)

    def emit(a, b, c):
        idx = len(code)
        code.append([a, b, c])
        return idx

    def emit_labeled(name, a, b, c):
        label(name)
        return emit(a, b, c)

    # Scratch cells
    sc = arr + n
    Z  = sc; mem[Z] = 0; sc += 1
    T1 = sc; sc += 1
    T2 = sc; sc += 1
    T3 = sc; sc += 1
    ONE = sc; mem[ONE] = 1; sc += 1
    NEG1 = sc; mem[NEG1] = -1; sc += 1
    OUTER = sc; sc += 1
    INNER = sc; sc += 1
    LIMIT = sc; sc += 1
    N_M1 = sc; mem[N_M1] = n - 1; sc += 1
    NEG_N_M1 = sc; mem[NEG_N_M1] = -(n - 1); sc += 1
    NEG_ARR = sc; mem[NEG_ARR] = -arr; sc += 1
    NEG_ARR_P1 = sc; mem[NEG_ARR_P1] = -(arr + 1); sc += 1
    ADDR_J = sc; sc += 1
    ADDR_J1 = sc; sc += 1
    # Negated address holders (set at runtime)
    NADDR_J = sc; sc += 1   # will hold -ADDR_J
    NADDR_J1 = sc; sc += 1  # will hold -ADDR_J1

    # ════════════════════════════════════════════════════════════
    # The program
    # ════════════════════════════════════════════════════════════

    # -- Outer loop init --
    emit_labeled('outer_init', OUTER, OUTER, 'inner_init')  # OUTER = 0

    # -- Inner loop init --
    emit_labeled('inner_init', INNER, INNER, '+')     # INNER = 0
    emit(LIMIT, LIMIT, '+')                           # LIMIT = 0
    emit(OUTER, LIMIT, '+')                           # LIMIT = -OUTER
    emit(NEG_N_M1, LIMIT, '+')                        # LIMIT = n-1-OUTER

    # -- Inner body: compute addresses --
    label('inner_body')
    emit(ADDR_J, ADDR_J, '+')
    emit(NEG_ARR, ADDR_J, '+')                        # ADDR_J = arr
    emit(T1, T1, '+')
    emit(INNER, T1, '+')                               # T1 = -INNER
    emit(T1, ADDR_J, '+')                              # ADDR_J = arr + INNER

    emit(ADDR_J1, ADDR_J1, '+')
    emit(NEG_ARR_P1, ADDR_J1, '+')                    # ADDR_J1 = arr + 1
    emit(T1, ADDR_J1, '+')                             # ADDR_J1 = arr+1+INNER

    # Compute negated addresses for patching
    emit(NADDR_J, NADDR_J, '+')
    emit(ADDR_J, NADDR_J, '+')                         # NADDR_J = -ADDR_J
    emit(NADDR_J1, NADDR_J1, '+')
    emit(ADDR_J1, NADDR_J1, '+')                       # NADDR_J1 = -ADDR_J1

    # -- Patch load instructions --
    # Each patch: clear target word, then subtract negated addr into it.
    # We'll use instruction-index-based references.
    # "patch_a(label)" means: clear mem[addr_of(label)], then mem[addr_of(label)] -= NADDR

    # We need to know the memory address of the 'a' field of instruction 'load_j'.
    # That's just 3 * instruction_index. We'll resolve this after laying out all code.
    # For now, we emit using placeholder scratch cells and fix up later.

    # Actually, it's easier to just emit the patch code targeting specific cell addresses.
    # We'll compute those addresses after code layout. For now, let's use a different
    # approach: instead of patching code cells, use INDIRECT addressing via known cells.
    #
    # Alternative approach: instead of self-modifying code, implement the compare-swap
    # using a helper subroutine that reads from ADDR_J and ADDR_J1 cells.
    # But SUBLEQ can only address cells by literal address in the instruction.
    # So we MUST use self-modifying code.
    #
    # Let me use a two-pass approach:
    # Pass 1: emit all code with placeholder addresses (0) for patched fields
    # Pass 2: compute the actual addresses and emit patch code

    # Load arr[j] into T1
    emit(T1, T1, '+')
    load_j = emit(0, T1, '+')          # a-field will be patched to ADDR_J

    # Negate: T3 = arr[j]
    emit(T3, T3, '+')
    emit(T1, T3, '+')                   # T3 = -T1 = arr[j]

    # Load arr[j+1] into T2
    emit(T2, T2, '+')
    load_j1 = emit(0, T2, '+')         # a-field will be patched to ADDR_J1

    # Negate: T1 = arr[j+1]
    emit(T1, T1, '+')
    emit(T2, T1, '+')                   # T1 = -T2 = arr[j+1]

    # Compare: T3 = arr[j] - arr[j+1]
    emit(T1, T3, '+')                   # T3 -= T1 = arr[j] - arr[j+1]

    # Branch: if T3 <= 0 (arr[j] <= arr[j+1]), skip swap
    emit(Z, T3, 'no_swap')

    # -- Swap --
    # Reload arr[j] into T3 (negated)
    emit(T3, T3, '+')
    reload_j = emit(0, T3, '+')         # T3 = -arr[j], a-field patched

    # Clear arr[j] and store arr[j+1]
    # T2 = -arr[j+1] from load above. But T2 was cleared by "emit(T2,T2,'+')".
    # Actually T2 = -arr[j+1] was set by the load. But then T1 was set from T2.
    # T2 is STILL -arr[j+1]. ✓ (SUBLEQ "emit(T2, T1, '+')" does T1 -= T2, doesn't touch T2.)
    # Wait: "emit(T1, T1, '+')" clears T1 to 0. Then "emit(T2, T1, '+')" does T1 -= T2 = -T2 = arr[j+1].
    # So T2 = -arr[j+1] still. ✓

    clr_j = emit(0, 0, '+')             # a,b patched to ADDR_J: clears arr[j]
    store_j = emit(T2, 0, '+')          # b patched to ADDR_J: arr[j] -= T2 = arr[j+1] ✓

    # Clear arr[j+1] and store arr[j]
    clr_j1 = emit(0, 0, '+')            # a,b patched to ADDR_J1: clears arr[j+1]
    store_j1 = emit(T3, 0, '+')         # b patched to ADDR_J1: arr[j+1] -= T3 = arr[j] ✓

    # -- No-swap target --
    label('no_swap')

    # Inner increment
    emit(NEG1, INNER, '+')               # INNER += 1

    # Inner test: if INNER < LIMIT, loop
    emit(T1, T1, '+')
    emit(LIMIT, T1, '+')                 # T1 = -LIMIT
    emit(T2, T2, '+')
    emit(T1, T2, '+')                    # T2 = LIMIT
    emit(INNER, T2, '+')                 # T2 = LIMIT - INNER
    emit(Z, T2, 'outer_inc')            # if LIMIT-INNER <= 0, exit inner loop
    emit(Z, Z, 'inner_body')            # else continue inner loop

    # Outer increment
    emit_labeled('outer_inc', NEG1, OUTER, '+')  # OUTER += 1

    emit(T1, T1, '+')
    emit(N_M1, T1, '+')                  # T1 = -(n-1)
    emit(T2, T2, '+')
    emit(T1, T2, '+')                    # T2 = n-1
    emit(OUTER, T2, '+')                 # T2 = n-1-OUTER
    emit(Z, T2, 'halt')                  # if n-1-OUTER <= 0, halt
    emit(Z, Z, 'inner_init')             # else next outer pass

    emit_labeled('halt', Z, Z, -1)

    # ════════════════════════════════════════════════════════════
    # Layout: convert code list to memory, resolve labels
    # ════════════════════════════════════════════════════════════

    # First, compute the instruction addresses for patch targets
    # load_j 'a' field = 3 * load_j
    # clr_j 'a' field = 3 * clr_j, 'b' field = 3 * clr_j + 1
    # store_j 'b' field = 3 * store_j + 1
    # etc.

    # Now insert patch code BEFORE the load instructions.
    # We need to splice patch instructions into the code list.
    # Easier approach: build a separate patch code section at the start of inner_body,
    # BEFORE the loads. This means we need to rearrange.

    # Actually, the simplest approach: add patch code before the loads.
    # The loads are at index `load_j` in the code list.
    # We can insert patch instructions at position `load_j - 2` (after addr computation).

    # Hmm, inserting into the middle of the code list would invalidate all indices.
    # Better approach: emit patch code at the BEGINNING of inner_body, BEFORE addr computation.
    # But we need ADDR_J/ADDR_J1 computed first...

    # The cleanest approach: have the patch code AFTER addr computation but BEFORE loads.
    # We already emitted in that order! The patch code should go between
    # the NADDR computation and the loads. Let me just insert it there.

    # Find the insertion point: it's right after NADDR_J1 computation,
    # before the T1 clear that precedes load_j.
    # That's at index `load_j - 1` (the "emit(T1, T1, '+')" before load_j).

    insert_at = load_j - 1  # before the first load setup

    # Build patch instructions
    patches = []

    def add_patch_a(instr_idx, naddr_cell):
        """Patch the 'a' field (offset 0) of instruction instr_idx."""
        target = 3 * instr_idx  # memory address of 'a' field
        patches.append([target, target, '+'])      # clear
        patches.append([naddr_cell, target, '+'])   # target -= -ADDR = ADDR

    def add_patch_b(instr_idx, naddr_cell):
        """Patch the 'b' field (offset 1) of instruction instr_idx."""
        target = 3 * instr_idx + 1
        patches.append([target, target, '+'])
        patches.append([naddr_cell, target, '+'])

    def add_patch_ab(instr_idx, naddr_cell):
        """Patch both 'a' and 'b' fields."""
        add_patch_a(instr_idx, naddr_cell)
        add_patch_b(instr_idx, naddr_cell)

    # BUT: the instr_idx values will shift after we insert the patches!
    # This is the chicken-and-egg problem.

    # Solution: compute the shift. We're inserting N patch instructions
    # at position insert_at. All instruction indices >= insert_at shift by N.
    # All indices < insert_at stay the same.

    # Count patches needed:
    # load_j 'a': 2 instr
    # load_j1 'a': 2 instr
    # reload_j 'a': 2 instr
    # clr_j 'a' + 'b': 4 instr
    # store_j 'b': 2 instr
    # clr_j1 'a' + 'b': 4 instr
    # store_j1 'b': 2 instr
    # Total: 18 instructions

    N_PATCH = 18
    shift = N_PATCH  # all indices >= insert_at shift by this much

    # Compute shifted indices
    def shifted(idx):
        return idx + shift if idx >= insert_at else idx

    s_load_j = shifted(load_j)
    s_load_j1 = shifted(load_j1)
    s_reload_j = shifted(reload_j)
    s_clr_j = shifted(clr_j)
    s_store_j = shifted(store_j)
    s_clr_j1 = shifted(clr_j1)
    s_store_j1 = shifted(store_j1)

    # Now build the actual patch instructions using shifted addresses
    # load_j 'a' field at memory address 3 * s_load_j
    def mk_patch_a(s_idx, naddr):
        t = 3 * s_idx
        return [[t, t, '+'], [naddr, t, '+']]

    def mk_patch_b(s_idx, naddr):
        t = 3 * s_idx + 1
        return [[t, t, '+'], [naddr, t, '+']]

    patches = []
    patches.extend(mk_patch_a(s_load_j, NADDR_J))
    patches.extend(mk_patch_a(s_load_j1, NADDR_J1))
    patches.extend(mk_patch_a(s_reload_j, NADDR_J))
    patches.extend(mk_patch_a(s_clr_j, NADDR_J))
    patches.extend(mk_patch_b(s_clr_j, NADDR_J))
    patches.extend(mk_patch_b(s_store_j, NADDR_J))
    patches.extend(mk_patch_a(s_clr_j1, NADDR_J1))
    patches.extend(mk_patch_b(s_clr_j1, NADDR_J1))
    patches.extend(mk_patch_b(s_store_j1, NADDR_J1))

    assert len(patches) == N_PATCH

    # Insert patches into code
    code[insert_at:insert_at] = patches

    # Update labels: shift those at or after insert_at
    for name in labels:
        if labels[name] >= insert_at:
            labels[name] += N_PATCH

    # Also update the instruction-index-based variables
    # (load_j etc. were pre-shift; now they're post-shift via s_*)

    # ════════════════════════════════════════════════════════════
    # Resolve labels and lay into memory
    # ════════════════════════════════════════════════════════════

    if len(code) * 3 > CODE_SIZE:
        raise ValueError(f"Code overflow: {len(code)} instructions = {len(code)*3} "
                         f"words > {CODE_SIZE}")

    for i, instr in enumerate(code):
        for f in range(3):
            val = instr[f]
            if val == '+':
                instr[f] = (i + 1) * 3  # "next instruction"
            elif isinstance(val, str):
                if val not in labels:
                    raise ValueError(f"Unknown label '{val}' in instruction {i}")
                instr[f] = labels[val] * 3  # label -> memory address

        mem[3*i], mem[3*i+1], mem[3*i+2] = instr[0], instr[1], instr[2]

    return mem, 0, arr, n

    # Scratch cells after the array
    sc = arr + n
    Z     = sc; mem[Z] = 0; sc += 1
    ONE   = sc; mem[ONE] = 1; sc += 1
    NEG1  = sc; mem[NEG1] = -1; sc += 1
    OUTER = sc; mem[OUTER] = 0; sc += 1
    INNER = sc; mem[INNER] = 0; sc += 1
    LIMIT = sc; mem[LIMIT] = n - 1; sc += 1
    T1    = sc; sc += 1   # temp
    T2    = sc; sc += 1   # temp
    T3    = sc; sc += 1   # temp
    # Constants for address computation
    NEG_N_M1    = sc; mem[NEG_N_M1] = -(n - 1); sc += 1
    NEG_ARR     = sc; mem[NEG_ARR] = -arr; sc += 1
    NEG_ARR_P1  = sc; mem[NEG_ARR_P1] = -(arr + 1); sc += 1
    N_M1        = sc; mem[N_M1] = n - 1; sc += 1
    # Self-modifying targets: these cells hold addresses that get written into code
    ADDR_J  = sc; sc += 1  # will hold arr + j
    ADDR_J1 = sc; sc += 1  # will hold arr + j + 1

    pc = 0
    def emit(a_, b_, c_):
        nonlocal pc
        if pc + 2 >= CODE_SIZE:
            raise ValueError(f"Code overflow at pc={pc}, CODE_SIZE={CODE_SIZE}")
        mem[pc], mem[pc + 1], mem[pc + 2] = a_, b_, c_
        ret = pc
        pc += 3
        return ret

    # ═══ OUTER LOOP INIT ═══
    # outer_init:
    outer_init = emit(OUTER, OUTER, pc + 3)    # OUTER = 0

    # ═══ INNER LOOP INIT ═══
    # Compute LIMIT = n - 1 - OUTER
    inner_init = emit(INNER, INNER, pc + 3)    # INNER = 0
    emit(LIMIT, LIMIT, pc + 3)                 # LIMIT = 0
    emit(OUTER, LIMIT, pc + 3)                 # LIMIT = -OUTER
    emit(NEG_N_M1, LIMIT, pc + 3)              # LIMIT -= -(n-1) => LIMIT = n-1-OUTER

    # ═══ INNER LOOP BODY ═══
    inner_body = pc

    # Step 1: Compute ADDR_J = arr + INNER, ADDR_J1 = arr + INNER + 1
    emit(ADDR_J, ADDR_J, pc + 3)               # ADDR_J = 0
    emit(NEG_ARR, ADDR_J, pc + 3)              # ADDR_J = arr
    emit(T1, T1, pc + 3)                        # T1 = 0
    emit(INNER, T1, pc + 3)                     # T1 = -INNER
    emit(T1, ADDR_J, pc + 3)                    # ADDR_J = arr + INNER

    emit(ADDR_J1, ADDR_J1, pc + 3)              # ADDR_J1 = 0
    emit(NEG_ARR_P1, ADDR_J1, pc + 3)          # ADDR_J1 = arr + 1
    emit(T1, ADDR_J1, pc + 3)                   # ADDR_J1 = arr + 1 + INNER
    # (T1 = -INNER is still valid)

    # Step 2: Self-modify — patch the 'a' field of load instructions.
    # We need to patch 4 instruction slots:
    #   LOAD_J:   reads arr[j]    (a-field)
    #   LOAD_J1:  reads arr[j+1]  (a-field)
    #   CLR_J:    clears arr[j]   (both a and b fields)
    #   CLR_J1:   clears arr[j+1] (both a and b fields)
    # Plus 2 store instructions' b-fields:
    #   STORE_J:  writes to arr[j]   (b-field)
    #   STORE_J1: writes to arr[j+1] (b-field)
    #
    # Each patch: clear target word, then subtract -ADDR into it.
    # 4 instructions per patch, 6 patches = 24 instructions = 72 words.
    # But we can share the negation: compute T2 = -ADDR_J, T3 = -ADDR_J1 once.

    # Compute T2 = -ADDR_J
    emit(T2, T2, pc + 3)
    emit(ADDR_J, T2, pc + 3)                    # T2 = -ADDR_J

    # Compute T3 = -ADDR_J1
    emit(T3, T3, pc + 3)
    emit(ADDR_J1, T3, pc + 3)                   # T3 = -ADDR_J1

    # We'll record the PCs of instructions to patch, then patch them.
    # But the instructions don't exist yet! Classic chicken-and-egg.
    # Solution: emit patches first (writing to known future PCs),
    # then emit the target instructions at those exact PCs.
    #
    # Alternative: reserve space for patches, emit targets, come back to fill patches.
    # Let's reserve space.

    patch_start = pc
    # 9 patches × 2 instructions each = 18 instr = 54 words.
    pc += 54  # reserve

    # Step 3: Load arr[j] and arr[j+1]
    emit(T1, T1, pc + 3)                         # T1 = 0
    LOAD_J = pc
    emit(0, T1, pc + 3)                           # T1 -= arr[j] => T1 = -arr[j]
    # (a-field of LOAD_J will be patched to ADDR_J)

    emit(T2, T2, pc + 3)                          # T2 = 0
    # Wait — T2 was -ADDR_J. We need it for patching but also for loading.
    # Let me use different temp strategy: T1 for arr[j], T2 for arr[j+1].
    # Actually, after patching T2/T3 were used to hold -ADDR values.
    # By this point the patches have run, so T2/T3 can be reused.
    # But wait: if we're reserving space for patches and THEN emitting loads,
    # the patches write T2→target at runtime. After patches run, T2 is modified
    # by SUBLEQ (b -= a). Let me trace through carefully.
    #
    # Actually, the SUBLEQ "emit(T2, target, next)" does: target -= T2.
    # It modifies target (which we want) AND checks if target <= 0 (branch).
    # T2 itself is NOT modified. So T2 = -ADDR_J is preserved across patches. ✓
    # Same for T3.
    #
    # BUT: the first patch does "emit(target, target, next)" which clears target.
    # That modifies target (a code cell), not T2. ✓
    #
    # So at Step 3, T2 = -ADDR_J and T3 = -ADDR_J1 are still valid.
    # But I just used T1 for the load. Let me redo:

    # Scrap — let me redo Steps 3-6 cleanly using only T1 as temp.
    pc = patch_start + 36  # after reserved patches

    # Load arr[j] into T1: T1 = 0, then SUBLEQ arr[j], T1, next => T1 = -arr[j]
    emit(T1, T1, pc + 3)
    LOAD_J = pc
    emit(0, T1, pc + 3)                           # T1 = -arr[j] (a-field patched)

    # Load arr[j+1] into T2: T2 is currently -ADDR_J, need to clear first.
    # Actually, we can just re-clear T2:
    emit(T2, T2, pc + 3)                          # T2 = 0
    LOAD_J1 = pc
    emit(0, T2, pc + 3)                           # T2 = -arr[j+1] (a-field patched)

    # Step 4: Compare arr[j] vs arr[j+1]
    # We have T1 = -arr[j], T2 = -arr[j+1].
    # diff = arr[j] - arr[j+1] = -T1 - (-T2) = T2 - T1.
    # Compute T3 = T2 - T1:
    emit(T3, T3, pc + 3)                          # T3 = 0
    emit(T1, T3, pc + 3)                          # T3 = -T1 = arr[j]
    emit(T2, T3, pc + 3)                          # T3 -= T2. T2=-arr[j+1], so T3 = arr[j] - (-arr[j+1])
    # Wait: T2 = -arr[j+1]. SUBLEQ T2, T3: T3 -= T2 = T3 - (-arr[j+1]) = arr[j] + arr[j+1]. WRONG.
    # Need T3 = arr[j] - arr[j+1].
    # T3 = -T1 = arr[j] (from above).
    # Now need to subtract arr[j+1]. We have T2 = -arr[j+1].
    # Negate T2: but that costs 3 instructions. Or compute differently.
    #
    # Better: T3 = 0, T3 -= T2 => T3 = arr[j+1]. Then T1 -= T3 ... no, modifies T1.
    # Simplest: T3 = arr[j] (= -T1). Then: need arr[j+1] to subtract.
    # T2 = -arr[j+1]. SUBLEQ (-T2), T3 ... we don't have -T2 directly.
    #
    # Let me just do: compute diff = arr[j+1] - arr[j] = -T2 - (-T1) = T1 - T2.
    # Compute in T3:
    emit(T3, T3, pc + 3)                          # T3 = 0
    # Oops: I already emitted 3 instructions for T3 above. Let me restart Step 4.
    # This is getting messy. Let me think about it more carefully.

    # CLEANER STEP 4:
    # We have T1 = -arr[j], T2 = -arr[j+1].
    # We want: diff = arr[j+1] - arr[j]. If diff <= 0, arr[j+1] <= arr[j], swap.
    # diff = arr[j+1] - arr[j] = (-T2) - (-T1) = T1 - T2.
    #
    # To compute T1 - T2 into T3:
    #   T3 = 0
    #   T3 -= T2  => T3 = -T2 = arr[j+1]    ... NO. SUBLEQ T2, T3: T3 -= T2 = -(-arr[j+1]) = arr[j+1]. YES.
    #   T3 -= (-T1) ... we need +T1.  SUBLEQ T1, T3: T3 -= T1 = arr[j+1] - (-arr[j]) = arr[j+1] + arr[j]. WRONG.
    #
    # Hmm. The issue is we have NEGATED values in T1, T2.
    # Let me use the original values approach instead.
    #
    # SIMPLEST CORRECT APPROACH:
    #   T3 = 0; T3 -= arr[j+1] (via patched load); T3 -= (-arr[j]) = T3 + arr[j]
    #   Wait, I only have the loads for T1, T2 which are patched.
    #
    # OK let me just use an extra temp and negate T1:
    #   T3 = 0; T3 -= T1; => T3 = arr[j].
    #   T3 -= (-T2)... still same problem.
    #
    # The real issue: SUBLEQ only does subtraction. To compute a - b,
    # I need b in a cell, then SUBLEQ b, result. With negated values,
    # SUBLEQ (-arr[j]), result subtracts (-arr[j]) from result, i.e., adds arr[j].
    #
    # So to compute diff = arr[j+1] - arr[j]:
    #   T3 = 0
    #   SUBLEQ T1, T3, next: T3 -= T1 = T3 - (-arr[j]) = arr[j]    ← adds arr[j] to T3. Now T3 = arr[j].
    #   SUBLEQ T2, T3, next: T3 -= T2 = T3 - (-arr[j+1]) = arr[j] + arr[j+1]   ← ADDS arr[j+1]!
    #
    # That gives arr[j] + arr[j+1], not arr[j+1] - arr[j]. SUBLEQ is SUBTRACT.
    #
    # CORRECT: to get arr[j+1] - arr[j], I subtract arr[j] from arr[j+1].
    # Start with T3 = arr[j+1], then SUBLEQ (cell_holding_arr[j]), T3 => T3 -= arr[j] = arr[j+1] - arr[j].
    # But I don't have arr[j] in a cell! I have -arr[j] in T1.
    # Fix: negate T1 into a temp cell. That costs 3 instructions.
    #
    # Actually: I have arr[j+1] - arr[j] = -(T2) - (-(T1)) = -T2 + T1.
    # This is: result starts at 0, subtract T2 (adds arr[j+1]), subtract (-T1) ...
    # No. Let me just do it step by step.
    #
    # Step 4 (correct):
    #   T3 = 0                                    (1 instr)
    #   T3 -= T1  => T3 = -T1 = arr[j]           (1 instr)   [SUBLEQ T1, T3, next]
    # Now T3 = arr[j]. Need to subtract arr[j+1]:
    # We have T2 = -arr[j+1]. Need to add T2 (which subtracts arr[j+1]).
    #   T3 -= (-T2) = T3 + T2 = arr[j] + (-arr[j+1]) = arr[j] - arr[j+1]
    # But SUBLEQ subtracts, so SUBLEQ T2, T3 => T3 -= T2 = arr[j] - (-arr[j+1]) = arr[j] + arr[j+1]
    # WRONG SIGN. SUBLEQ T2, T3 does T3 = T3 - T2 = arr[j] - (-arr[j+1]) = arr[j] + arr[j+1].
    #
    # The fundamental issue: I want T3 = arr[j] - arr[j+1].
    # T3 = arr[j] after step above.
    # SUBLEQ (cell holding arr[j+1]), T3 => T3 -= arr[j+1] ✓
    # But I have -arr[j+1] in T2, not arr[j+1].
    # So: negate T2 into another temp, then subtract.
    #
    # OK final approach using 4 temps: T1=-arr[j], T2=-arr[j+1]. Negate both.

    # Let me back up the PC and re-emit step 3+4 cleanly.
    # I'll undo the bad emits above and start over from after the patch reservation.

    pc = patch_start + 36

    # ── Load arr[j] ──
    emit(T1, T1, pc + 3)
    LOAD_J = pc
    emit(0, T1, pc + 3)               # T1 = -arr[j]  (a patched to ADDR_J)

    # ── Negate: T3 = arr[j] ──
    emit(T3, T3, pc + 3)              # T3 = 0
    emit(T1, T3, pc + 3)              # T3 -= T1 = arr[j]

    # ── Load arr[j+1] ──
    emit(T2, T2, pc + 3)
    LOAD_J1 = pc
    emit(0, T2, pc + 3)               # T2 = -arr[j+1]  (a patched to ADDR_J1)

    # ── Negate: T1_pos = arr[j+1] (reuse T1) ──
    emit(T1, T1, pc + 3)              # T1 = 0
    emit(T2, T1, pc + 3)              # T1 -= T2 = arr[j+1]

    # Now T3 = arr[j], T1 = arr[j+1].
    # Compute diff = T3 - T1 = arr[j] - arr[j+1].
    # SUBLEQ T1, T3, next: T3 -= T1 = arr[j] - arr[j+1].
    emit(T1, T3, pc + 3)              # T3 = arr[j] - arr[j+1]

    # ── Branch: if diff <= 0 (arr[j] <= arr[j+1]), skip swap ──
    # SUBLEQ Z, T3, no_swap: T3 -= 0, unchanged. If T3 <= 0, goto no_swap.
    BRANCH = pc
    emit(Z, T3, 0)                    # target patched below

    # ═══ SWAP arr[j] and arr[j+1] ═══
    # Clear arr[j], store arr[j+1] there. Clear arr[j+1], store arr[j] there.
    # We still have T1 = arr[j+1] from above, but T3 was modified by the branch.
    # Reload? No — T1 and T2 still hold arr[j+1] and -arr[j+1].
    # Actually T1 = arr[j+1] still, T2 = -arr[j+1] still. And we need arr[j].
    # T3 = arr[j] - arr[j+1] (modified). But we can reconstruct:
    # arr[j] = T3 + T1 = (arr[j]-arr[j+1]) + arr[j+1] = arr[j]. ✓
    # But T3 was also modified by SUBLEQ Z, T3: T3 -= 0 = T3. Unchanged! ✓
    # So T3 = arr[j] - arr[j+1], T1 = arr[j+1].
    # Actually we need the original arr[j]. T3 + T1 would work but requires addition.
    # Simpler: just use T2 = -arr[j+1]. Negate it back.
    #
    # Actually, we just need to write. We have:
    #   T1 = arr[j+1]  (to store into arr[j])
    #   T2 = -arr[j+1] (not needed)
    #   T3 = arr[j] - arr[j+1] (not directly useful)
    # We need arr[j] to store into arr[j+1]. Reload it:

    # Re-load arr[j] from memory (one patched instruction):
    emit(T3, T3, pc + 3)              # T3 = 0
    RELOAD_J = pc
    emit(0, T3, pc + 3)               # T3 = -arr[j]  (a patched to ADDR_J)

    # Now write arr[j+1] (= T1) to position arr[j]:
    # Clear arr[j]: SUBLEQ arr[j], arr[j], next (self-subtract clears to 0)
    CLR_J = pc
    emit(0, 0, pc + 3)                # a,b both patched to ADDR_J

    # Store: arr[j] -= (-T1) = arr[j] + arr[j+1]. But arr[j] = 0 now. Wait:
    # SUBLEQ T1, arr[j], next: arr[j] -= T1 = 0 - arr[j+1] = -arr[j+1]. WRONG SIGN.
    # We need to store arr[j+1] = T1, but SUBLEQ subtracts.
    # T2 = -arr[j+1]. SUBLEQ T2, arr[j]: arr[j] -= T2 = 0 - (-arr[j+1]) = arr[j+1]. ✓
    STORE_J = pc
    emit(T2, 0, pc + 3)               # b patched to ADDR_J. arr[j] = arr[j+1]. ✓

    # Write arr[j] (= -T3 since T3 = -arr[j]) to position arr[j+1]:
    # Clear arr[j+1]:
    CLR_J1 = pc
    emit(0, 0, pc + 3)                # a,b both patched to ADDR_J1

    # Store: SUBLEQ T3, arr[j+1]: arr[j+1] -= T3 = 0 - (-arr[j]) = arr[j]. ✓
    STORE_J1 = pc
    emit(T3, 0, pc + 3)               # b patched to ADDR_J1. arr[j+1] = arr[j]. ✓

    # ═══ NO-SWAP LANDS HERE ═══
    NO_SWAP = pc
    mem[BRANCH + 2] = NO_SWAP         # patch branch target

    # ═══ INNER LOOP INCREMENT & TEST ═══
    emit(NEG1, INNER, pc + 3)         # INNER += 1

    # Test: if INNER < LIMIT, loop
    # Compute INNER - LIMIT. If <= 0, go to inner_body.
    emit(T1, T1, pc + 3)
    emit(LIMIT, T1, pc + 3)           # T1 = -LIMIT
    emit(T2, T2, pc + 3)
    emit(T1, T2, pc + 3)              # T2 = LIMIT
    emit(INNER, T2, pc + 3)           # T2 = LIMIT - INNER (note: backwards!)
    # If LIMIT - INNER <= 0 (INNER >= LIMIT), fall through to outer increment.
    # If LIMIT - INNER > 0 (INNER < LIMIT), pc += 3 → falls through. WRONG.
    # We want: if INNER < LIMIT, goto inner_body.
    # SUBLEQ Z, T2, target: T2 unchanged. If T2 <= 0 (LIMIT <= INNER), goto target.
    # We want the OPPOSITE: goto inner_body when INNER < LIMIT (T2 > 0).
    # Solution: after SUBLEQ Z, T2, exit_inner, fall through = continue loop.
    # Then add unconditional jump to inner_body.
    emit(Z, T2, pc + 6)               # if LIMIT-INNER <= 0, skip ahead (exit inner)
    emit(Z, Z, inner_body)            # else: unconditional jump to inner_body

    # ═══ OUTER LOOP INCREMENT & TEST ═══
    emit(NEG1, OUTER, pc + 3)         # OUTER += 1

    # Test: if OUTER < n-1, continue
    emit(T1, T1, pc + 3)
    emit(N_M1, T1, pc + 3)            # T1 = -(n-1)
    emit(T2, T2, pc + 3)
    emit(T1, T2, pc + 3)              # T2 = n-1
    emit(OUTER, T2, pc + 3)           # T2 = n-1 - OUTER
    emit(Z, T2, pc + 6)               # if n-1-OUTER <= 0, skip (exit outer)
    emit(Z, Z, inner_init)            # else: goto inner_init

    # ═══ HALT ═══
    emit(Z, Z, -1)

    # ═══ FILL IN PATCHES (back-patch the reserved space) ═══
    save_pc = pc
    pc = patch_start

    # Re-compute T2 = -ADDR_J and T3 = -ADDR_J1 before using them as patch sources.
    # Actually, T2 and T3 were computed BEFORE the reserved patch space.
    # At runtime, the code flows: compute T2, T3 → patches → loads → compare → swap → loop.
    # So when patches execute, T2 = -ADDR_J and T3 = -ADDR_J1 are still set. ✓
    # (The patches are the first thing to use T2/T3 after they're set.)

    # Patch LOAD_J 'a' field: mem[LOAD_J] = ADDR_J
    emit(LOAD_J, LOAD_J, pc + 3)      # clear
    emit(T2, LOAD_J, pc + 3)          # mem[LOAD_J] -= T2 = mem[LOAD_J] - (-ADDR_J) = ADDR_J ✓

    # Patch LOAD_J1 'a' field: mem[LOAD_J1] = ADDR_J1
    emit(LOAD_J1, LOAD_J1, pc + 3)
    emit(T3, LOAD_J1, pc + 3)         # ADDR_J1 ✓

    # Patch RELOAD_J 'a' field: mem[RELOAD_J] = ADDR_J
    emit(RELOAD_J, RELOAD_J, pc + 3)
    emit(T2, RELOAD_J, pc + 3)        # ADDR_J ✓

    # Patch CLR_J 'a' and 'b' fields: mem[CLR_J] = mem[CLR_J+1] = ADDR_J
    emit(CLR_J, CLR_J, pc + 3)
    emit(T2, CLR_J, pc + 3)           # 'a' ✓
    clr_j_b = CLR_J + 1
    emit(clr_j_b, clr_j_b, pc + 3)
    emit(T2, clr_j_b, pc + 3)         # 'b' ✓

    # Patch STORE_J 'b' field: mem[STORE_J+1] = ADDR_J
    store_j_b = STORE_J + 1
    emit(store_j_b, store_j_b, pc + 3)
    emit(T2, store_j_b, pc + 3)       # ✓

    # Patch CLR_J1 'a' and 'b' fields: ADDR_J1
    emit(CLR_J1, CLR_J1, pc + 3)
    emit(T3, CLR_J1, pc + 3)
    clr_j1_b = CLR_J1 + 1
    emit(clr_j1_b, clr_j1_b, pc + 3)
    emit(T3, clr_j1_b, pc + 3)

    # Patch STORE_J1 'b' field: ADDR_J1
    store_j1_b = STORE_J1 + 1
    emit(store_j1_b, store_j1_b, pc + 3)
    emit(T3, store_j1_b, pc + 3)

    # That's 18 instructions = 54 words. We reserved 36. Check:
    used = pc - patch_start
    if used > 54:
        raise ValueError(f"Patch overflow: used {used} words, reserved 54")

    # Pad remaining reserved space
    while pc < patch_start + 54:
        emit(Z, Z, pc + 3)

    pc = save_pc
    return mem, 0, arr, n


def make_random_program(n_instr=None, seed=None):
    """Generate a random SUBLEQ program for stress testing.

    Returns (memory, pc). Run with the interpreter and compare.
    """
    if seed is not None:
        random.seed(seed)
    if n_instr is None:
        n_instr = random.randint(3, 20)

    mem = _fresh_mem()
    # Random instructions
    for i in range(min(n_instr, CODE_SIZE // 3)):
        base = i * 3
        mem[base] = random.randint(0, MEM_SIZE - 1)      # a
        mem[base + 1] = random.randint(0, MEM_SIZE - 1)   # b
        mem[base + 2] = random.randint(-1, min(n_instr * 3, CODE_SIZE - 1))  # c
    # Random data
    for i in range(DATA_START, MEM_SIZE):
        mem[i] = random.randint(VALUE_MIN // 100, VALUE_MAX // 100)

    return mem, 0
