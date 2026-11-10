import argparse
import json
from pathlib import Path
from .volume import Dataset
from .session import Session


def main():
    ap = argparse.ArgumentParser(description="Agent-controlled connectomics workbench")
    sub = ap.add_subparsers(dest="command", required=True)
    demo = sub.add_parser(
        "demo",
        help="Generate a synthetic fixture and scripted, replayable demonstration",
    )
    demo.add_argument("--out", type=Path, required=True)
    for name in ["serve", "run", "stdio"]:
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--out", required=True)
        if name == "serve":
            p.add_argument("--port", type=int, default=8096)
            p.add_argument("--neuroglancer", action="store_true")
        if name == "run":
            p.add_argument(
                "--provider", choices=["responses", "anthropic", "chat", "openrouter"], default="responses"
            )
            p.add_argument("--model", required=True)
            p.add_argument("--model-options", type=json.loads, default={})
            p.add_argument("--watch-port", type=int)
            p.add_argument("--endpoint")
            p.add_argument("--max-turns", type=int, default=100)
            p.add_argument("--max-actions", type=int, default=500)
            p.add_argument("--max-output-tokens", type=int, default=8192)
    p = sub.add_parser("replay")
    p.add_argument("--dataset", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("export")
    p.add_argument("--session", required=True)
    p = sub.add_parser("freeze-evaluation")
    p.add_argument("--dataset", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--scope", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--dataset", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("import-world")
    p.add_argument("--world", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--bounds",
        type=int,
        nargs=6,
        required=True,
        metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
    )
    args = ap.parse_args()
    if args.command == "demo":
        from .demo import create, scripted
        from .replay import export, replay
        from .evaluate import freeze, evaluate_frozen

        dataset = create(args.out / "data")
        freeze(
            Dataset.load(dataset),
            args.out / "data/evaluation/reference.npy",
            args.out / "data/evaluation/scope.npy",
            args.out / "evaluation-spec.json",
        )
        session = scripted(dataset, args.out / "episode")
        report = evaluate_frozen(
            session.dataset, session.output, args.out / "evaluation-spec.json"
        )
        (args.out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "evaluation": report,
                    "replay": replay(
                        session.dataset, session.output, args.out / "verified-replay"
                    ),
                    "exports": export(session.output),
                },
                indent=2,
            )
        )
    elif args.command == "import-world":
        from .import_world import import_world

        print(import_world(args.world, args.out, args.bounds))
    elif args.command == "export":
        from .replay import export

        print(json.dumps(export(args.session), indent=2))
    elif args.command == "freeze-evaluation":
        from .evaluate import freeze

        freeze(Dataset.load(args.dataset), args.reference, args.scope, args.out)
        print("Evaluation reference and scope frozen: " + args.out)
    elif args.command == "evaluate":
        from .evaluate import evaluate_frozen

        report = evaluate_frozen(Dataset.load(args.dataset), args.session, args.spec)
        with Path(args.out).open("x") as f:
            f.write(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    elif args.command == "replay":
        from .replay import replay

        print(
            json.dumps(
                replay(Dataset.load(args.dataset), args.session, args.out), indent=2
            )
        )
    else:
        session = Session(Dataset.load(args.dataset), args.out)
        if args.command == "serve":
            from .server import make_server

            server = make_server(session, args.port)
            print(f"Workbench: http://127.0.0.1:{server.server_port}", flush=True)
            print("Local API bearer token: " + server.agent_token, flush=True)
            bridge = None
            if args.neuroglancer:
                from .neuroglancer_bridge import NeuroglancerBridge

                bridge = NeuroglancerBridge(session, server.session_lock)
                print("Native Neuroglancer: " + bridge.url, flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                if bridge:
                    bridge.close()
                server.server_close()
        elif args.command == "run":
            from .harness import run

            from .server import live_observer

            with live_observer(session, args.watch_port) as observer_url:
                if observer_url:
                    print("Read-only live observer: " + observer_url, flush=True)
                print(
                    json.dumps(
                        run(
                            session,
                            args.model,
                            args.provider,
                            args.max_turns,
                            args.max_actions,
                            args.max_output_tokens,
                            args.endpoint,
                            model_options=args.model_options,
                        ),
                        indent=2,
                    )
                )
        else:
            import sys
            from .tools import schemas

            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    result = (
                        {"tools": schemas()}
                        if request.get("name") == "tools"
                        else session.execute(
                            request["name"], request.get("arguments", {})
                        )
                    )
                except (ValueError, TypeError, KeyError) as exc:
                    result = {"ok": False, "error": str(exc)}
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
