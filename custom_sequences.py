import random
from typing import Generator, Callable


def generate_bounded_sequence(n: int, bound: int) -> Generator[int, None, None]:
    for _ in range(n):
        yield random.randint(0, bound)

def sequence(value_func: Callable[[int], float]) -> Generator[float, None, None]:
    i = 0
    while True:
        yield value_func(i)
        i += 1