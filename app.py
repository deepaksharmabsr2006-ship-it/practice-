"""
Vercel FastAPI Entrypoint.
Exposes the ASGI 'app' instance at repository root for Vercel deployment.
"""

from backend.app import app

# Export for Vercel
__all__ = ["app"]
