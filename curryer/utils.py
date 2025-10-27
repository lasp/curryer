"""Generic utilities (not specific to the SPICE library).

@author: Brandon Stone
"""

import datetime
import logging
import logging.config
import subprocess
import time
import typing
from pathlib import Path

logger = logging.getLogger(__name__)


def track_performance(func: typing.Callable, storage: dict = None):
    """Wrapper to track the performance of functions or class methods.

    Parameters
    ----------
    func : callable
        Function to track.
    storage : dict, optional
        Dictionary to store the metrics in, otherwise it is assumed that func is
        part of an instance and metrics will be added to the
        `_performance_metrics` attribute.

    Returns
    -------
    callable
        Wrapped function or class method.

    """
    label = func.__qualname__

    def _timed_func(*args, **kwargs):
        self = None
        if storage is None:
            self = args[0]
            if not hasattr(self, "_performance_metrics"):
                setattr(self, "_performance_metrics", {})

        t0 = time.time()
        output = func(*args, **kwargs)
        td = time.time() - t0

        metrics = (storage if self is None else self._performance_metrics).setdefault(
            label,
            {
                "count": 0,
                "total": 0.0,
                "min": 9e99,
                "max": -1.0,
            },
        )
        metrics["count"] += 1
        metrics["total"] += td
        metrics["min"] = min(td, metrics["min"])
        metrics["max"] = max(td, metrics["max"])

        return output

    return _timed_func


def format_performance(obj, indent: str = "\t", p: int = 3, ascending=None):
    """Format the performance results from `track_performance`.

    Parameters
    ----------
    obj : instance or dict
        Dictionary of the performance metrics or instance with the automated
        `_performance_metrics` attribute.
    indent : str, optional
        String used for indentation. Default is tab.
    p : int, optional
        Decimal precision in timing format. Default is 3, aka milliseconds.
    ascending : None or bool, optional
        Order results by insert (None), ascending (True), or descending (False).
        Default is None.

    Returns
    -------
    str
        Formatted performance results.

    """
    if isinstance(obj, dict):
        metrics = obj
    else:
        metrics = getattr(obj, "_performance_metrics", None)
        if metrics is None:
            raise ValueError(f"Unable to report performance metrics missing for: {obj}")

    totals = {}
    results = {}
    for label, stats in metrics.items():
        txt = f"{indent}{label}"
        for cat in ("count", "total", "min", "mean", "max"):
            txt += f"\n{indent * 2}{cat:>5}: "
            if cat == "count":
                txt += f"{stats['count']:3}"
            elif cat == "mean":
                txt += f"{stats['total'] / stats['count']:{4 + p}.{p}f}"
            else:
                txt += f"{stats[cat]:{4 + p}.{p}f}"
        totals[label] = stats["total"]
        results[label] = txt

    if ascending is True:
        order = sorted(totals, key=lambda key: totals[key])
    elif ascending is False:
        order = reversed(sorted(totals, key=lambda key: totals[key]))
    else:
        order = list(totals)

    txt = f"Performance Summary (ascending={ascending}):"
    for label in order:
        txt += f"\n{results[label]}"
    return txt


def capture_subprocess(cmd, timeout=3600, capture_output=False):
    """Execute commands in a subprocess.

    Parameters
    ----------
    cmd : list of str
        Command arguments to execute.
    timeout : int, optional
        Number of seconds to wait before timing out. Default=3600 (1hr)
    capture_output : bool, optional
        Option to return the stdout text. Default=False

    Returns
    -------
    None or str
        The stdout text is returned if `capture_output`=True.

    """

    def log_pipe_output(obj, lgr, level, msg=""):
        """Check if pipe output can be logged, and do it."""
        if obj is not None and obj != b"":
            lgr.log(level, "%s:\n%s", msg, obj.decode().rstrip())

    logger.debug("Executing subprocess command: %r", " ".join(cmd))
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            timeout=timeout,
            check=True,
            stdout=subprocess.PIPE if logger.isEnabledFor(logging.DEBUG) or capture_output else None,
            stderr=subprocess.PIPE if logger.isEnabledFor(logging.ERROR) else None,
        )
    except subprocess.CalledProcessError as e:
        logger.error("Error code returned: %i", e.returncode)
        log_pipe_output(e.output, logger, logging.WARNING, msg="Stdout")
        log_pipe_output(e.stderr, logger, logging.ERROR, msg="Stderr")
        logger.exception("Exception:")
        raise
    else:
        log_pipe_output(proc.stdout, logger, logging.DEBUG, msg="Stdout")
        log_pipe_output(proc.stderr, logger, logging.WARNING, msg="Stderr")
        logger.debug("Subprocess completed. Return code: %i", proc.returncode)

    if capture_output:
        return proc.stdout.decode()
    return


def enable_logging(
    log_level=logging.DEBUG, log_file: typing.Union[bool, str, Path] = False, extra_loggers: list[str] = None
):
    """Enable logging to the console and optionally to a file.

    Parameters
    ----------
    log_level : int
        A logging log level.
    log_file : bool or str or Path, optional
        Option to enable logging to a file. If true or a directory the filename
        will be auto-generated. If true, the file will be saved to the current
        working directory. Otherwise, the supplied file will be used.
    extra_loggers : List[str], optional
        Collection of additional loggers to enable at DEBUG level.

    """
    root_level = "DEBUG" if log_file else logging.getLevelName(log_level)
    extra_loggers = {} if not extra_loggers else {name: {"level": root_level} for name in extra_loggers}

    # Configure script logging.
    log_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "simple": {
                "class": "logging.Formatter",
                "format": "[%(asctime)s.%(msecs)03d] %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
            "detailed": {
                "class": "logging.Formatter",
                "format": "[%(asctime)s.%(msecs)03d %(name)s.%(funcName)s:%(lineno)i %(levelname)5.5s] %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {"class": "logging.NullHandler"},
        },
        "loggers": dict(
            **{
                "curryer": {"level": root_level},
                "cdflib": {"level": "ERROR"},
            },
            **extra_loggers,
        ),
        "root": {"level": root_level, "handlers": ["console", "file"]},
    }

    # Optionally, add logging to a file.
    if log_file:
        default_log_name = f"curryer.{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.log"  # TODO[rename]
        if log_file is True:
            log_file = Path.cwd() / default_log_name
        else:
            log_file = Path(log_file)
            if log_file.is_dir():
                log_file = log_file / default_log_name

        log_config["handlers"]["file"] = {
            "level": root_level,
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(log_file),
            "mode": "a",
        }

    # Initialize and configure the loggers.
    logging.config.dictConfig(log_config)

    if log_file:
        logger.debug("Logging to file: %s", log_file)
    return
