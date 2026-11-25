# Production entry point: runs the app with the waitress server.
import os

from dotenv import load_dotenv

load_dotenv()  # read settings from a .env file if there is one

from waitress import serve  # noqa: E402
from app import app  # noqa: E402

host = os.environ.get("HOST", "127.0.0.1")
port = int(os.environ.get("PORT", "5001"))
print(f"Serving on http://{host}:{port} with waitress")
serve(app, host=host, port=port)