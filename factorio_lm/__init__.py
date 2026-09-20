"""FactorioLM model training and exact integer reference inference."""

from .int32 import INT32_MAX, INT32_MIN, add, mul, sub, trunc_div, wrap

__all__ = [
    "INT32_MAX",
    "INT32_MIN",
    "add",
    "mul",
    "sub",
    "trunc_div",
    "wrap",
]

__version__ = "0.2.0"
