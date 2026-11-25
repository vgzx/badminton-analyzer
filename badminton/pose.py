"""姿态后端。

三档，按可用性自动降级，报告里会明确写出实际用的是哪一档：

* ``mediapipe``：MediaPipe Pose（33 点），质量最好，CPU 也能跑；
* ``proxy``：没有姿态模型时，只用球员框的几何量（身高、框顶、框宽变化）
  作为“身体姿态的代理”。**它不是真姿态**，只能看蹲伏与起跳的相对变化，
  报告里必须标注为代理指标；
* ``off``：完全不做姿态。

所有关节坐标统一返回**全帧像素坐标**，这样后续做速度/角度时不受裁剪框抖动影响。
"""

from __future__ import annotations

import math

# MediaPipe Pose 关键点索引
IDX = {
    "nose": 0,
    "shoulder_l": 11, "shoulder_r": 12,
    "elbow_l": 13, "elbow_r": 14,
    "wrist_l": 15, "wrist_r": 16,
    "hip_l": 23, "hip_r": 24,
    "knee_l": 25, "knee_r": 26,
    "ankle_l": 27, "ankle_r": 28,
}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle_at(p, a, b):
    """以 p 为顶点，pa 与 pb 的夹角（度）。"""
    v1 = (a[0] - p[0], a[1] - p[1])
    v2 = (b[0] - p[0], b[1] - p[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _derive(joints, height_px):
    """从关键点推导姿态指标。"""
    out = {
        "mode": "mediapipe",
        "height_px": float(height_px),
        "torso_px": None,
        "shoulder_px": None,
        "crouch": None,
        "stance": None,
        "racket_up": None,
        "wrist_l": joints.get("wrist_l"),
        "wrist_r": joints.get("wrist_r"),
        "elbow_angle_l": None,
        "elbow_angle_r": None,
    }
    need = ("shoulder_l", "shoulder_r", "hip_l", "hip_r", "ankle_l", "ankle_r",
            "elbow_l", "elbow_r", "wrist_l", "wrist_r")
    if any(key not in joints for key in need):
        return out

    sh_mid = _mid(joints["shoulder_l"], joints["shoulder_r"])
    hip_mid = _mid(joints["hip_l"], joints["hip_r"])
    ankle_mid = _mid(joints["ankle_l"], joints["ankle_r"])
    torso = _dist(sh_mid, hip_mid)
    shoulder = _dist(joints["shoulder_l"], joints["shoulder_r"])
    out["torso_px"] = torso
    out["shoulder_px"] = shoulder

    if torso > 3.0:
        # 直立时约 1.7~2.1；下蹲时变小
        out["crouch"] = (ankle_mid[1] - hip_mid[1]) / torso
    if shoulder > 3.0:
        out["stance"] = _dist(joints["ankle_l"], joints["ankle_r"]) / shoulder
    out["racket_up"] = bool(min(joints["wrist_l"][1], joints["wrist_r"][1]) < sh_mid[1])
    out["elbow_angle_l"] = _angle_at(joints["elbow_l"], joints["shoulder_l"], joints["wrist_l"])
    out["elbow_angle_r"] = _angle_at(joints["elbow_r"], joints["shoulder_r"], joints["wrist_r"])
    out["shoulder_mid"] = sh_mid
    return out


class MediaPipePose:
    name = "mediapipe"

    def __init__(self, model_complexity=0, min_detection=0.4, min_tracking=0.4):
        import mediapipe as mp  # noqa: WPS433

        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=int(model_complexity),
            smooth_landmarks=True,
            min_detection_confidence=float(min_detection),
            min_tracking_confidence=float(min_tracking),
        )

    def infer(self, crop_bgr, offset=(0, 0)):
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        rgb = crop_bgr[:, :, ::-1]
        try:
            result = self._pose.process(rgb)
        except Exception:
            return None
        if not result.pose_landmarks:
            return None
        h, w = crop_bgr.shape[:2]
        ox, oy = offset
        joints = {}
        for key, idx in IDX.items():
            lm = result.pose_landmarks.landmark[idx]
            if getattr(lm, "visibility", 1.0) < 0.35:
                continue
            joints[key] = (ox + lm.x * w, oy + lm.y * h)
        if "shoulder_l" not in joints or "shoulder_r" not in joints:
            return None
        return _derive(joints, h)

    def close(self):
        try:
            self._pose.close()
        except Exception:
            pass


class ProxyPose:
    """没有姿态模型时的几何代理：只用球员框。"""

    name = "proxy"

    def infer(self, bbox, height_px=None):
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        h = float(y2 - y1)
        return {
            "mode": "proxy",
            "height_px": h,
            "torso_px": None,
            "shoulder_px": None,
            "crouch": None,
            "stance": None,
            "racket_up": None,
            "wrist_l": None,
            "wrist_r": None,
            "elbow_angle_l": None,
            "elbow_angle_r": None,
            "top_y": float(y1),
            "width_px": float(x2 - x1),
        }

    def close(self):
        pass


class PoseService:
    """对外统一入口：负责后端选择 + 降级。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = None
        self.mode = "off"
        wanted = (cfg.pose_backend or "auto").lower()

        if wanted in ("off", "none", "0"):
            return
        if wanted in ("mediapipe", "auto"):
            try:
                self.backend = MediaPipePose()
                self.mode = "mediapipe"
                return
            except Exception as exc:
                if wanted == "mediapipe":
                    raise RuntimeError(
                        "指定了 mediapipe 姿态后端但加载失败：%s\n"
                        "安装：pip install mediapipe" % exc)
        # auto 或 proxy 都落到代理模式
        self.backend = ProxyPose()
        self.mode = "proxy"

    @property
    def available(self) -> bool:
        return self.backend is not None

    @property
    def is_real_pose(self) -> bool:
        return self.mode == "mediapipe"

    def infer(self, crop_bgr, bbox, offset=(0, 0)):
        if self.backend is None:
            return None
        if self.mode == "mediapipe":
            return self.backend.infer(crop_bgr, offset=offset)
        return self.backend.infer(bbox)

    def close(self):
        if self.backend is not None:
            self.backend.close()
