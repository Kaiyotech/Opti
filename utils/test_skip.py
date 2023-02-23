import numpy as np


def test(x, selector_skip_k):
    tick = 0
    count = 0
    ticks = []
    for i in range(x):
        do_selector = do_selector_action(selector_skip_k, tick) if selector_skip_k is not None else True
        if do_selector:
            ticks.append(tick)
        tick = 0 if do_selector else tick + 1
        count += 1 if do_selector else 0
    print(x * 4 / (120 * count))
    ticks = np.asarray(ticks)
    mean = np.mean(ticks) * (4 / 120)
    std = np.std(ticks) * (4 / 120)
    print(f"mean: {mean}     std: {std}")


def do_selector_action(selector_skip_k, tick) -> bool:
    p = 1 / (1 + (selector_skip_k * tick))
    if np.random.uniform() < p:
        return False
    else:
        return True
