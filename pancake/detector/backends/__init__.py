from .backend import Backend
from .backend_dei import DEI
from .backend_simple import SIMPLE

BACKEND_REGISTRY = Backend.get_subclasses()
