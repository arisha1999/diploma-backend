#!/usr/bin/env python3
import logging
import multiprocessing
import gunicorn.app.base
from os import environ, path
from gunicorn.six import iteritems

from app import app
from app.utils.log import StubbedGunicornLogger, configure_logs


def number_of_workers():
    return (multiprocessing.cpu_count() * 2) + 1


class FlaskApplication(gunicorn.app.base.BaseApplication):

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super(FlaskApplication, self).__init__()

    def load_config(self):
        config = dict([(key, value) for key, value in iteritems(self.options)
                       if key in self.cfg.settings and value is not None])
        for key, value in iteritems(config):
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


if __name__ == '__main__':
    configure_logs(filename=path.basename(__file__).split('.')[0] + '.log')

    options = {
        'bind': '%s:%s' % (environ.get('APP_HOST'), environ.get('APP_PORT')),
        'workers': number_of_workers(),
        'timeout': int(environ.get('WORKER_TTL')),
        "logger_class": StubbedGunicornLogger
    }
    FlaskApplication(app, options).run()

