"""Vercel Function entry point: every /api/* route except the waitlist is rewritten here."""

from src.api.main import app  # noqa: F401
