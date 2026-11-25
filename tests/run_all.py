"""测试运行器：python -m tests.run_all（或 python tests/run_all.py）。

不依赖 pytest，也不依赖 OpenCV —— 只验证纯逻辑（标定、状态机、解码、指标、报告）。
"""

from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODULES = ["test_court", "test_rally", "test_hits", "test_metrics", "test_report"]


def main():
    passed = failed = 0
    failures = []
    for name in MODULES:
        module = __import__(name)
        for attr in sorted(dir(module)):
            if not attr.startswith("test_"):
                continue
            func = getattr(module, attr)
            if not callable(func):
                continue
            label = "%s.%s" % (name, attr)
            try:
                func()
                print("  PASS  %s" % label)
                passed += 1
            except Exception as exc:
                print("  FAIL  %s -> %s: %s" % (label, type(exc).__name__, exc))
                failures.append((label, traceback.format_exc()))
                failed += 1
    print("\n%d passed, %d failed" % (passed, failed))
    for label, tb in failures:
        print("\n--- %s ---\n%s" % (label, tb))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
