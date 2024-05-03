"""utils

@author: Brandon Stone
"""
import inspect
import logging
from functools import wraps

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def find_mapping_path(map_, from_='ugps', to='ugps'):
    """Find a chain of items that link two keys (breadth search).
    """
    if from_ not in map_:
        raise ValueError(f'Unsupported time format from_=[{from_}]! Valid: {list(map_)}')
    if to not in map_:
        raise ValueError(f'Unsupported time format to=[{to}]! Valid: {list(map_)}')

    options = [[from_]]
    found = None

    while not found and options:
        todo = []
        for steps in options:
            src = steps[-1]

            for cur in map_[src]:
                if cur in steps and cur != to:
                    continue

                nextup = steps + [cur]
                if cur == to:
                    found = nextup
                    break

                todo.append(nextup)
            if found:
                break
        options = todo

    if not found or found[0] != from_ or found[-1] != to:
        raise ValueError(f'Unable to find path from [{from_}] to [{to}] in map=[{map_}]!')

    rendered = []
    for i in range(len(found) - 1):
        rendered.append(map_[found[i]][found[i + 1]])

    if logger.isEnabledFor(5):
        logger.log(5, 'Found mapping for [%s] -> [%s] as steps=[%s] with maps=[%s]', from_, to, found, rendered)

    return rendered


def apply_conversions(funcs, times, /, **kwargs):
    """Apply a set of time conversions.
    """
    for func in funcs:
        times = func(times, **kwargs)
    return times


class InputAsArray:
    """Decorator to guarantee the first argument is a non-scalar numpy array.
    """

    def __init__(self, dtype, filter_keywords=True, defaults=None):
        """Numpy data type of the array.

        Parameters
        ----------
        dtype : str or numpy.dtype
            Numpy data type to convert the array to if it differs.

        """
        self.dtype = dtype
        self.filter_keywords = filter_keywords
        self.defaults = defaults

    def __call__(self, func):
        """Method wrapper.

        Parameters
        ----------
        func : callable
            Method to wrap. First arugment `times` will be converted to a numpy
            array with data type `dtype`.

        Returns
        -------
        callable
            Method whose first argument `times` is converted to a non-scalar
            numpy array of `dtype`.

        """
        spec = inspect.getfullargspec(func)

        @wraps(func)
        def internal(times, *args, **kwargs):
            """Wrapped method.
            """
            was_scalar = False

            # Handle scalars.
            if np.isscalar(times):
                was_scalar = True
                times = np.array([times], dtype=self.dtype)

            # Handle non-array iterable.
            elif not isinstance(times, (np.ndarray, pd.Series)):
                times = np.array(times, dtype=self.dtype)

            # Handle arrays of the wrong type.
            elif times.dtype != self.dtype:
                times = times.astype(self.dtype)

            # Force 1d array
            orig_shape = None
            if times.ndim > 1:
                orig_shape = times.shape
                times = times.ravel()

            if self.defaults:
                for key, val in self.defaults.items():
                    kwargs.setdefault(key, val)

            if self.filter_keywords and kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k in spec.args}

            # Apply the function to the standardized input.
            output = func(times, *args, **kwargs)

            # Undo the shape and scaler handling.
            if orig_shape is not None:
                output = output.reshape(orig_shape)
            return output[0] if was_scalar else output

        return internal


@InputAsArray(np.float64)
def noop_float64(times):
    """No-op float64 type.
    """
    return times


@InputAsArray(np.int64)
def noop_int64(times):
    """No-op int64 type.
    """
    return times


@InputAsArray(np.str_)
def noop_str(times):
    """No-op string type.
    """
    return times


@InputAsArray('M8[us]')
def noop_dt64(times):
    """No-op datetime type.
    """
    return times
