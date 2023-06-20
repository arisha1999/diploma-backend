from flask import Flask
from flask_cors import CORS
from flask_session import Session

class FlaskApplication:
    def __init__(self):
        self.static_folder = '../public/dist/'
        self.app = Flask(__name__, static_folder=self.static_folder)

        self.__route_list = []
        self.__cors_instance = CORS()
        self.__session_instance = Session()


    def __unpack_routes(self, routes=None, prefix=''):

        if not routes:
            from app.routes import ROUTES
            routes = ROUTES

        for route in routes:
            route_prefix = prefix + route.get('prefix', '')
            if route.get('group'):
                self.__unpack_routes(routes=route.get('group'), prefix=route_prefix)
            if route.get('controller'):
                self.__route_list.append({
                    'blueprint': route['controller'],
                    'url_prefix': route_prefix
                })

    def __register_blueprints(self):
        for route in self.__route_list:
            self.app.register_blueprint(**route)


    def init(self):
        self.__cors_instance.init_app(self.app)
        self.__session_instance.init_app(self.app)

        self.__unpack_routes()
        self.__register_blueprints()

        return self.app
