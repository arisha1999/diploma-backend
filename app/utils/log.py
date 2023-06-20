import logging
from os import environ
from loguru import logger

from gunicorn.glogging import Logger

LOG_LEVEL = logging.getLevelName(environ.get("LOG_LEVEL", "DEBUG"))

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class StubbedGunicornLogger(Logger):
    def setup(self, cfg):
        handler = logging.NullHandler()
        self.error_logger = logging.getLogger("gunicorn.error")
        self.error_logger.addHandler(handler)
        self.access_logger = logging.getLogger("gunicorn.access")
        self.access_logger.addHandler(handler)
        self.error_log.setLevel(LOG_LEVEL)
        self.access_log.setLevel(LOG_LEVEL)


def configure_logs(filename, log_level=LOG_LEVEL, **kwargs):
    intercept_handler = InterceptHandler()
    logging.root.setLevel(log_level)
    logging.getLogger().handlers = [intercept_handler]

    seen = set()
    for name in [
        *logging.root.manager.loggerDict.keys(),
        'STDOUT',
        'STDERR',
        "gunicorn",
        "gunicorn.access",
        "gunicorn.error",
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
    ]:
        if name not in seen:
            seen.add(name.split(".")[0])
            logging.getLogger(name).handlers = [intercept_handler]

    default_handler_conf = {
        'sink': environ.get('LOG_PATH') + '/{filename}'.format(filename=filename),
        'backtrace': False,
        'diagnose': False,
        'rotation': '1 week',
        'enqueue': True,
        'retention': 4,
        'compression': 'zip'
    }
    default_handler_conf.update(kwargs)
    logger.configure(handlers=[default_handler_conf])