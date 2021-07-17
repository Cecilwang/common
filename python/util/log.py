import logging

_loggers = {}


def logger(name, level=logging.INFO):
    if name not in _loggers:
        fmt = "%(asctime)s-%(levelname)s-%(name)s : %(message)s"
        fmt = logging.Formatter(fmt)
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        log = logging.getLogger(name)
        log.addHandler(handler)
        log.setLevel(level)
        _loggers[name] = log
    return _loggers[name]
