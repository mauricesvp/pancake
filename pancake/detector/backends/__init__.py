from .backend import Backend
from .backend_dei import DEI

BACKEND_REGISTRY = Backend.get_subclasses()
