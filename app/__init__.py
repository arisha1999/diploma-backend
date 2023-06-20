import os
import time
from flask import jsonify, send_from_directory
from app.application import FlaskApplication
from werkzeug.exceptions import HTTPException

from app.utils.utils import debug
from app.exceptions import AppException

time.tzset()
builder = FlaskApplication()
app = builder.init()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


@app.errorhandler(AppException)
def handle_invalid_usage(error):
    code = 400
    if isinstance(error, HTTPException):
        code = error.code
    return jsonify({
        'message': str(error),
        'exception': error.__class__.__name__
    }), code
