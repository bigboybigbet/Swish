"""
Run the web app:

    python -m server                      # http://127.0.0.1:8000, opens a browser
    python -m server --reload             # dev auto-reload; still 1 worker
    python -m server --port 8010 --no-browser

NBA_HOST / NBA_PORT (environment or .env) override the defaults; the flags
override both. NBA_QUANT_DEMO_DATE=YYYY-MM-DD replays a past date offline.
"""
import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from server import clock, config_store, paths, security  # noqa: F401  (paths: mimetypes fix)

    env_host, env_port = config_store.server_address()       # reads .env
    ap = argparse.ArgumentParser(prog="python -m server", description="NBA Quant web app")
    ap.add_argument("--host", default=env_host, help=f"bind address (default {env_host})")
    ap.add_argument("--port", type=int, default=env_port, help=f"port (default {env_port})")
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args(argv)

    try:
        demo = clock.demo_date()
    except ValueError as e:
        ap.error(str(e))

    # create_app() runs in uvicorn (and in the reload worker): pass the address via env.
    os.environ["NBA_HOST"], os.environ["NBA_PORT"] = args.host, str(args.port)

    if not security.is_loopback_host(args.host):
        print(f"WARNING: binding to {args.host}, not loopback. Anyone who can reach this "
              "address can use the app; secret and delete endpoints still accept "
              "loopback clients only.", file=sys.stderr)

    browse_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{browse_host}:{args.port}"
    print(f"NBA Quant at {url}" + (f"  (demo mode: replaying {demo})" if demo else ""))
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()

    import uvicorn
    uvicorn.run(
        "server.app:create_app", factory=True,
        host=args.host, port=args.port, workers=1,
        reload=args.reload,
        reload_dirs=[str(ROOT / d) for d in ("server", "engine", "data")] if args.reload else None,
    )


if __name__ == "__main__":
    main()
