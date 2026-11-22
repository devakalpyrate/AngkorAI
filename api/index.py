import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app

class VercelPathMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        forwarded_uri = environ.get('HTTP_X_FORWARDED_URI') or environ.get('HTTP_X_MATCHED_PATH')
        if forwarded_uri and forwarded_uri not in ('/api/index', '/api/index.py'):
            parts = forwarded_uri.split('?', 1)
            path = parts[0]
            environ['PATH_INFO'] = path
            environ['REQUEST_URI'] = forwarded_uri
            if len(parts) > 1 and not environ.get('QUERY_STRING'):
                environ['QUERY_STRING'] = parts[1]
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathMiddleware(app.wsgi_app)
