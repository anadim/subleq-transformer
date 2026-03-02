"""
SUBLEQ Transformer — a neural network that learned to be a computer.

A 4.9M-param transformer trained on single-step SUBLEQ execution that
generalizes to arbitrary multi-step programs (Fibonacci, multiplication,
division, square root) never seen during training.
"""

from .interpreter import (
    MEM_SIZE, BYTES_PER_VALUE, VOCAB_SIZE, VALUE_MIN, VALUE_MAX,
    CODE_SIZE, DATA_START, SEQ_LEN,
    clamp, step, run,
)
from .tokenizer import (
    value_to_bytes, bytes_to_value,
    encode, decode, get_changed_positions,
)
from .programs import (
    make_negate, make_addition, make_countdown, make_multiply,
    make_fibonacci, make_div, make_isqrt, make_halt,
    generate_random_program, generate_random_state,
)
from .data import (
    generate_step_pair, generate_trace_pairs,
    generate_batch, generate_trace_batch, pregenerate_data,
)
from .model import MiniSUBLEQTransformer
