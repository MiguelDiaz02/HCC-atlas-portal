"""
WSGI entry point for Render deployment.
Imports the Dash app and exposes its Flask server.
"""
from app import server

if __name__ == "__main__":
    server.run(debug=False, host="0.0.0.0", port=5000)
