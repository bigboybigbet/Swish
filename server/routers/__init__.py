"""
Router auto-registration.

Every module in this package that defines a module-level `router`
(fastapi.APIRouter) is included by create_app(); a new milestone only drops
a file here. Optional module attributes:

    ORDER = 100            # inclusion order (lower first; pages.py uses 1000)
    def on_startup(app):   # called once when the app starts
    def on_shutdown(app):  # called once when the app stops

Import errors are not swallowed: a broken router fails startup loudly.
"""
import importlib
import pkgutil


def discover() -> list:
    """Router modules in inclusion order."""
    modules = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        if getattr(mod, "router", None) is not None:
            modules.append(mod)
    return sorted(modules, key=lambda m: (getattr(m, "ORDER", 100), m.__name__))
