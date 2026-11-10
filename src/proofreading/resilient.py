"""Bounded, checkpoint-preserving supervisor for portable experiments."""
import argparse
import fcntl
from pathlib import Path
import time

from .experiment import execute
from .recovery import atomic_json, failure_state


def supervise(config, output, *, resume_from=None, attempts=3, retry_delay=30,
              transport=None, sleep=time.sleep, resume_latest=False):
    if attempts < 1 or retry_delay < 0:
        raise ValueError('Invalid recovery limits')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Supervisor already active') from None
        # Existing runs require an explicit resume source: never guess and rebill.
        existing = sorted(p for p in output.glob('attempt-[0-9]*') if p.is_dir())
        if resume_latest and resume_from is not None:
            raise ValueError('Choose resume_latest or resume_from, not both')
        if resume_latest and existing:
            resume_from = existing[-1]
        if existing and resume_from is None:
            raise ValueError('Existing attempts require --resume-from')
        source = resume_from
        for i in range(attempts):
            target = output / f'attempt-{len(existing)+i+1:03d}'
            atomic_json(output / 'status.json', {'status': 'running', 'attempt': str(target),
                        'resume_from': str(source) if source else None})
            try:
                result = execute(config, target, transport=transport, resume_from=source)
            except Exception as exc:
                state = failure_state(exc)
                atomic_json(output / 'status.json', {**state, 'attempt': str(target),
                            'resume_command_source': str(target),
                            'attempts_remaining': attempts-i-1})
                if not state['retryable'] or i+1 == attempts:
                    raise
                # A timeout can have incurred a provider charge; only missing
                # responses are resubmitted. Already saved tool calls are rebuilt
                # in a fresh local session, never applied twice to the same one.
                source = target
                sleep(min(300, retry_delay * 2**i))
            else:
                atomic_json(output / 'status.json', {'status': 'completed', 'attempt': str(target)})
                return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--resume-latest', action='store_true', help='On scheduler restart, recover the last saved attempt')
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--retry-delay', type=float, default=30)
    a = p.parse_args()
    supervise(a.config, a.out, resume_from=a.resume_from, attempts=a.attempts, retry_delay=a.retry_delay, resume_latest=a.resume_latest)


if __name__ == '__main__':
    main()
