import inspect
from algorithms.deepinversion import distill_deep_inversion
from algorithms.data_free import train_dfad

def run_distillation(
    algorithm_name: str,
    teacher,
    student,
    train_loader,
    test_loader=None,
    config=None
):
    if config is None:
        config = {}

    supported_algorithms = {
        "dfad": train_dfad,
        "deepinv": distill_deep_inversion
    }

    if algorithm_name not in supported_algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Supported: {list(supported_algorithms.keys())}")

    train_func = supported_algorithms[algorithm_name]

    valid_params = inspect.signature(train_func).parameters

    unexpected_keys = [k for k in config if k not in valid_params]
    if unexpected_keys:
        raise TypeError(f"{algorithm_name} got unexpected config arguments: {unexpected_keys}")

    kwargs = config.copy()
    kwargs.update({
        'teacher': teacher,
        'student': student,
        'test_loader': test_loader,
    })

    return train_func(**kwargs)
