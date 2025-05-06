import inspect
import warnings

from algorithms.deepinversion import distill_deep_inversion
from algorithms.data_free import train_dfad


def run_distillation(
    algorithm_name: str,
    teacher,
    student,
    train_loader,
    test_loader=None,
    config=None,
):
    if config is None:
        config = {}

    supported_algorithms = {
        "dfad": train_dfad,
        "deepinv": distill_deep_inversion,
    }

    if algorithm_name not in supported_algorithms:
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {list(supported_algorithms.keys())}")

    algorithm_fn = supported_algorithms[algorithm_name]
    print(f"[INFO] Running algorithm: {algorithm_name}")
    
    sig = inspect.signature(algorithm_fn)
    accepted_args = list(sig.parameters.keys())

    base_args = {
        'teacher': teacher,
        'student': student,
        'train_loader': train_loader,
        'test_loader': test_loader
    }

    full_args = {**base_args, **config}

    filtered_args = {}
    for key, val in full_args.items():
        if key in accepted_args:
            filtered_args[key] = val
        else:
            warnings.warn(f"[WARNING] Argument '{key}' is not used by '{algorithm_name}' and will be ignored.")

    print(f"[INFO] Using arguments for '{algorithm_name}': {list(filtered_args.keys())}")
    
    return algorithm_fn(**filtered_args)
