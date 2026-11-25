"""测试辅助：所有临时产物都落在项目内 .tmp_tests/，避免被系统临时目录策略挡住。"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tmpdir(name):
    path = os.path.join(ROOT, ".tmp_tests", name)
    os.makedirs(path, exist_ok=True)
    return path
