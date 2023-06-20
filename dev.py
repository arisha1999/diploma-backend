#!/usr/bin/env python3
from app import app

if __name__ == '__main__':
    app.run(host='192.168.255.201', port=5000, debug=True, use_reloader=True, threaded=True)