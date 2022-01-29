from flask import Flask
import config
from apps.index.view import index_bp

def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.from_object(config.DeveplomentConfig)
    app.register_blueprint(index_bp)

    return app