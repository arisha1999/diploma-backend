import pprint
from flask import Response

def debug(variable):
    pp = pprint.PrettyPrinter(indent=4, depth=6)
    debug_message = pp.pprint(variable)
    return Response(debug_message, mimetype="text/text")
