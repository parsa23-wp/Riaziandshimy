"""Minimal test runner (used when pytest is not installed).

Usage: python run_tests.py [-k substring] [--slow]
Discovers ``test_*`` functions in ``tests/test_*.py`` (not tests/ui).
Functions with the attribute ``slow = True`` run only with --slow.
"""
from __future__ import annotations

import importlib
import pathlib
import sys
import time
import traceback


def main(argv: list[str]) -> int:
    key = argv[argv.index("-k") + 1] if "-k" in argv else ""
    slow = "--slow" in argv
    root = pathlib.Path(__file__).parent
    sys.path.insert(0, str(root))
    passed = failed = skipped = 0
    for f in sorted((root / "tests").glob("test_*.py")):
        mod = importlib.import_module(f"tests.{f.stem}")
        for name in sorted(dir(mod)):
            fn = getattr(mod, name)
            if not (name.startswith("test_") and callable(fn)) or key not in name:
                continue
            if getattr(fn, "slow", False) and not slow:
                skipped += 1
                continue
            t = time.time()
            try:
                fn()
                passed += 1
            except Exception:
                failed += 1
                print(f"FAIL {f.stem}::{name}")
                traceback.print_exc()
            else:
                if time.time() - t > 5:
                    print(f"slow ok {name} {time.time() - t:.1f}s")
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
