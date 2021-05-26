
def learning_scheduling(
    initial_value: float, iteration: int, decay: float=0.01
) -> float:
    return initial_value * (1 / (1 + decay * iteration))