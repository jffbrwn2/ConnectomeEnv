"""Standalone read-only episode viewer. No Sites, database, GPU or model key needed."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit, unquote
from .snapshots import snapshot


def make_server(root, port=8097):
    root = Path(root).resolve()
    episodes = {}
    for p in sorted(root.rglob('harness.json')):
        directory = p.parent.resolve()
        if directory.is_relative_to(root) and (directory / 'initial-observations.json').is_file():
            episodes[str(len(episodes))] = directory

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, value, mime='application/json'):
            data = json.dumps(value).encode() if mime == 'application/json' else value
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            host = self.headers.get('Host', '')
            if host not in (f'localhost:{self.server.server_port}', f'127.0.0.1:{self.server.server_port}'):
                return self.send(403, {'error': 'Invalid host; use an SSH tunnel'})
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + host:
                return self.send(403, {'error': 'Cross-origin request rejected'})
            path = unquote(urlsplit(self.path).path)
            if path == '/':
                return self.send(200, Path(__file__).with_name('results.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/episodes':
                return self.send(200, [{'id': key, 'name': str(directory.relative_to(root)),
                                       'summary': json.loads((directory / 'harness.json').read_text())}
                                      for key, directory in episodes.items()])
            parts = path.strip('/').split('/')
            if len(parts) >= 3 and parts[0] == 'episodes' and parts[1] in episodes:
                directory = episodes[parts[1]]
                if parts[2:] == ['snapshot']:
                    try:
                        data, _ = snapshot(directory)
                        result = directory.parent / 'result.json'
                        # Only scores, never evaluator data or full provider response logs.
                        data['scores'] = json.loads(result.read_text()).get('scores', {}) if result.exists() else {}
                        return self.send(200, data)
                    except (ValueError, KeyError, OSError) as exc:
                        return self.send(409, {'error': 'Episode is incomplete or invalid: ' + type(exc).__name__})
                if len(parts) == 4 and parts[2] == 'frames':
                    target = (directory / 'frames' / parts[3]).resolve()
                    if target.is_relative_to(directory / 'frames') and target.suffix == '.png' and target.is_file():
                        return self.send(200, target.read_bytes(), 'image/png')
            self.send(404, {'error': 'Not found'})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--port', type=int, default=8097)
    args = p.parse_args()
    server = make_server(args.results, args.port)
    print(f'Results: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
