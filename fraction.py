from typing import Generator, Sequence

def fraction_element(iterations: int = 1000, support_seq: Sequence[int] = (1,) ) -> Generator[float, None, None]:
    value = 0
    iteration = 0
    while iteration < iterations:
        try:
            item = support_seq[iteration]
        except IndexError:
            item = None
        support_seq_i = item if item else support_seq[-1]
        value = 1/(support_seq_i + value)
        yield value
        iteration += 1

def approximate_fraction(iterations: int = 1000, support_seq: Sequence[int] = (1,)) -> float:
    gen = fraction_element(iterations, support_seq)
    last_value = None
    for val in gen:
        last_value = val
    estimated = last_value
    return estimated
