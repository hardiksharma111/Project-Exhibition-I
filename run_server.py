"""
Server Launcher Script
Launches the FastAPI web server on http://localhost:8000
"""

import uvicorn
import sys

if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"
    print("=" * 65)
    print("   INTELLIGENT MULTI-MODEL AI: ADAPTIVE SLM ROUTING ENGINE")
    print(f"   Server listening at: http://{host}:{port}")
    print(f"   API Documentation:   http://{host}:{port}/docs")
    print("=" * 65)

    uvicorn.run("api.app:app", host=host, port=port, reload=False, log_level="info")
