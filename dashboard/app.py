"""
╔══════════════════════════════════════════════════════════════════════════╗
║  FLASK DASHBOARD — Application Factory + SocketIO                       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import logging
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

log = logging.getLogger(__name__)

socketio = SocketIO()


def create_app(config=None) -> Flask:
    """Create Flask application with SocketIO support."""
    cfg = config or {}

    app = Flask(__name__,
                template_folder="templates",
                static_folder="static")

    app.config["SECRET_KEY"] = cfg.get("secret_key", "smart-hospital-2026")

    CORS(app)

    # Register routes
    from dashboard.routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    # Initialize SocketIO
    socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

    # Register SocketIO events
    from dashboard.socketio_events import register_events
    register_events(socketio)

    log.info(f"Dashboard app created")
    return app
