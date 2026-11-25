"""OpenCV 懒加载。

设计原因：核心逻辑（标定、状态机、解码、报告）只依赖 numpy，可以在没有
OpenCV 的机器上跑自检与单元测试。只有真正处理视频的模块才需要 cv2。
"""

_cv2 = None
_error = None


def get_cv2():
    """返回 cv2 模块；不可用时抛出带安装指引的 RuntimeError。"""
    global _cv2, _error
    if _cv2 is not None:
        return _cv2
    if _error is not None:
        raise RuntimeError(_error)
    try:
        import cv2  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        _error = (
            "缺少 OpenCV。请先安装：\n"
            "    pip install opencv-python-headless\n"
            "（桌面环境也可以用 opencv-python）\n"
            "原始错误：%s" % exc
        )
        raise RuntimeError(_error) from exc
    _cv2 = cv2
    return _cv2


def has_cv2():
    try:
        get_cv2()
        return True
    except RuntimeError:
        return False
