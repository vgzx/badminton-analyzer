"""逐帧特征：几何 -> 平滑 -> 挥拍分量。

所有位置先经 :class:`~badminton.court.Court` 变换到球场单位 u，
速度因此和时间、分辨率解耦。

平滑策略：速度/能量用一阶 EMA（时间常数可配），
动作能量另有一条**慢速基线**，用 ``energy - baseline`` 作为“这一帧有多反常”，
这样既不会被整体亮度变化带偏，也能突出挥拍瞬间。
"""

from __future__ import annotations

import math

from .court import Court


def _ema(prev, value, dt, tau):
    if prev is None:
        return float(value)
    if tau <= 1e-6:
        return float(value)
    alpha = 1.0 - math.exp(-max(dt, 1e-6) / tau)
    return float(prev + alpha * (value - prev))


class PlayerFeat:
    __slots__ = (
        "present", "x", "y", "px", "py", "speed", "accel", "energy", "energy_hp",
        "height", "crouch", "stance", "racket_up", "wrist_v", "elbow_v",
        "comps", "pose_ok", "h_ref",
    )

    def __init__(self):
        self.present = False
        self.x = float("nan")
        self.y = float("nan")
        self.px = float("nan")
        self.py = float("nan")
        self.speed = 0.0
        self.accel = 0.0
        self.energy = 0.0
        self.energy_hp = 0.0
        self.height = float("nan")
        self.crouch = float("nan")
        self.stance = float("nan")
        self.racket_up = None
        self.wrist_v = 0.0
        self.elbow_v = 0.0
        self.comps = {}
        self.pose_ok = False
        self.h_ref = float("nan")

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


class FrameFeat:
    __slots__ = ("t", "idx", "dt", "near", "far", "state")

    def __init__(self, t, idx, dt, near, far, state=""):
        self.t = float(t)
        self.idx = int(idx)
        self.dt = float(dt)
        self.near = near
        self.far = far
        self.state = state


class _SideState:
    __slots__ = ("prev_x", "prev_y", "prev_t", "prev_speed", "prev_accel",
                 "prev_energy", "baseline", "prev_h", "h_ref",
                 "prev_wrist", "prev_elbow")

    def __init__(self):
        self.prev_x = None
        self.prev_y = None
        self.prev_t = None
        self.prev_speed = 0.0
        self.prev_accel = 0.0
        self.prev_energy = None
        self.baseline = None
        self.prev_h = None
        self.h_ref = None
        self.prev_wrist = {}
        self.prev_elbow = {}


class FeatureExtractor:
    def __init__(self, court: Court, cfg):
        self.court = court
        self.cfg = cfg
        self.sides = {"near": _SideState(), "far": _SideState()}

    # ------------------------------------------------------------------
    def _update_side(self, side, obs, t, dt, pose):
        st = self.sides[side]
        feat = PlayerFeat()

        if obs is None:
            # 丢失：保留上一次位置，速度衰减而不是瞬间归零
            if st.prev_x is not None:
                feat.x, feat.y = st.prev_x, st.prev_y
                feat.speed = _ema(st.prev_speed, 0.0, dt * 2.0, self.cfg.speed_tau)
                st.prev_speed = feat.speed
                st.prev_t = t
            return feat

        x_u, y_u = self.court.to_unit(obs.x, obs.y)
        feat.present = True
        feat.x, feat.y = x_u, y_u
        feat.px, feat.py = obs.x, obs.y
        feat.height = obs.h

        feat.energy = _ema(st.prev_energy, obs.energy, dt, self.cfg.energy_tau)
        st.prev_energy = feat.energy
        st.baseline = _ema(st.baseline, obs.energy, dt, self.cfg.baseline_tau)
        feat.energy_hp = feat.energy - (st.baseline if st.baseline is not None else feat.energy)

        # 站立高度基线：快速上探、缓慢回落
        if st.h_ref is None:
            st.h_ref = obs.h
        elif obs.h > st.h_ref:
            st.h_ref += 0.25 * (obs.h - st.h_ref)
        else:
            st.h_ref += 0.004 * (obs.h - st.h_ref)
        feat.h_ref = st.h_ref

        # 速度 / 加速度
        raw_speed = 0.0
        if st.prev_x is not None and st.prev_t is not None:
            gap = t - st.prev_t
            if 1e-3 < gap <= 1.0:
                raw_speed = math.hypot(x_u - st.prev_x, y_u - st.prev_y) / gap
        feat.speed = _ema(st.prev_speed, raw_speed, dt, self.cfg.speed_tau)
        raw_accel = (feat.speed - st.prev_speed) / max(dt, 1e-3)
        feat.accel = _ema(st.prev_accel, raw_accel, dt, self.cfg.speed_tau)
        st.prev_speed = feat.speed
        st.prev_accel = feat.accel
        st.prev_x, st.prev_y, st.prev_t = x_u, y_u, t

        # 姿态
        if pose:
            feat.pose_ok = pose.get("mode") == "mediapipe"
            if feat.pose_ok:
                if pose.get("crouch") is not None:
                    feat.crouch = pose["crouch"]
                if pose.get("stance") is not None:
                    feat.stance = pose["stance"]
                feat.racket_up = pose.get("racket_up")
                torso = pose.get("torso_px") or obs.h
                wv = ev = 0.0
                for arm in ("l", "r"):
                    wrist = pose.get("wrist_%s" % arm)
                    prev = st.prev_wrist.get(arm)
                    if wrist is not None and prev is not None and dt > 1e-3:
                        wv = max(wv, math.hypot(wrist[0] - prev[0], wrist[1] - prev[1])
                                 / dt / max(torso, 1.0))
                    if wrist is not None:
                        st.prev_wrist[arm] = wrist
                    angle = pose.get("elbow_angle_%s" % arm)
                    prev_a = st.prev_elbow.get(arm)
                    if angle is not None and prev_a is not None and dt > 1e-3:
                        ev = max(ev, abs(angle - prev_a) / dt)
                    if angle is not None:
                        st.prev_elbow[arm] = angle
                feat.wrist_v = wv
                feat.elbow_v = ev / 100.0
            else:
                # 代理模式：用框高相对站立基线的比值当“蹲伏代理”
                if st.h_ref and st.h_ref > 1:
                    feat.crouch = obs.h / st.h_ref

        # 挥拍分量（供 hits.py 做稳健 z 分数组合）
        height_rate = 0.0
        if st.prev_h is not None and dt > 1e-3:
            height_rate = abs(obs.h - st.prev_h) / dt / max(obs.h, 1.0)
        st.prev_h = obs.h
        feat.comps = {
            "accel": abs(feat.accel),
            "energy_hp": max(feat.energy_hp, 0.0),
            "wrist_v": feat.wrist_v,
            "height_rate": height_rate,
        }
        return feat

    # ------------------------------------------------------------------
    def update(self, t, idx, dt, obs_near, obs_far, pose_near=None):
        near = self._update_side("near", obs_near, t, dt, pose_near)
        far = self._update_side("far", obs_far, t, dt, None)
        return FrameFeat(t, idx, dt, near, far)
