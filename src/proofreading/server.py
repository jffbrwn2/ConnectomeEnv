"""Local browser/API workbench. Serves only public state and saved agent images."""

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit
from .tools import schemas


def make_server(session, port=8096, read_only=False):
    token = secrets.token_urlsafe(32)
    lock = session.lock
    actions = ThreadPoolExecutor(max_workers=1, thread_name_prefix="proofreading-tools")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, data, mime="application/json"):
            body = json.dumps(data).encode() if mime == "application/json" else data
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            host = self.headers.get("Host", "")
            expected = f"127.0.0.1:{self.server.server_port}"
            if host not in [expected, f"localhost:{self.server.server_port}"]:
                self.send(403, {"error": "Invalid host"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + host:
                self.send(403, {"error": "Cross-origin request rejected"})
                return False
            return True

        def authorized(self):
            value = self.headers.get("Authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(value, token):
                self.send(401, {"error": "Bearer token required"})
                return False
            return True

        def do_GET(self):
            if not self.allowed():
                return
            path = urlsplit(self.path).path
            if path == "/":
                page = (
                    Path(__file__)
                    .with_name("watch.html" if read_only else "workbench.html")
                    .read_text()
                    .replace("__TOKEN__", token)
                )
                return self.send(200, page.encode(), "text/html; charset=utf-8")
            if not self.authorized():
                return
            if path == "/api/tools":
                return self.send(200, schemas())
            if path == "/api/state":
                with lock:
                    return self.send(200, session.state())
            if path == "/api/events":
                with lock:
                    return self.send(200, session.events)
            if path.startswith("/frames/"):
                target = (session.output / path.lstrip("/")).resolve()
                if (
                    target.is_relative_to((session.output / "frames").resolve())
                    and target.suffix == ".png"
                    and target.is_file()
                ):
                    return self.send(200, target.read_bytes(), "image/png")
            self.send(404, {"error": "Unknown route"})

        def do_POST(self):
            if not self.allowed() or not self.authorized():
                return
            if read_only:
                return self.send(403, {"error": "Observer cannot change the episode"})
            if self.path != "/api/tool":
                return self.send(404, {"error": "Unknown route"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("Request must be 1 byte to 1 MiB")
                body = json.loads(self.rfile.read(size))
                # A stable worker keeps its OpenGL context and GPU buffers
                # across HTTP connections. Session.execute owns the state lock.
                result = actions.submit(session.execute,body["name"],body.get("arguments",{})).result()
                self.send(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.send(400, {"error": str(exc)})

    class Server(ThreadingHTTPServer):
        def server_close(self):
            super().server_close()
            actions.shutdown(wait=True,cancel_futures=True)

    server = Server(("127.0.0.1", port), Handler)
    server.agent_token = token
    server.session_lock = lock
    return server


from contextlib import contextmanager


@contextmanager
def live_observer(session, port):
    if port is None:
        yield None
        return
    server = make_server(session, port, read_only=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
