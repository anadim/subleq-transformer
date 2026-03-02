"""Round 1: Hand-coded SUBLEQ transformer with analytically set weights."""

from .interpreter import (
    MEM_SIZE, CODE_SIZE, DATA_START, VALUE_MIN, VALUE_MAX,
    VALUE_OFFSET, VOCAB_SIZE, SEQ_LEN,
    clamp, step, run,
)
from .model import HandCodedSUBLEQ
