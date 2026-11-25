"""合成特征流：用于自检与单元测试（不需要视频、不需要 OpenCV）。

它模拟一场“发球准备 -> 活球对拉 -> 静止 -> 走动捡球”的节奏，
并按固定节拍在**交替的一方**身上打出挥拍尖峰，因此可以验证：
    回合切分是否抓到正确区间、拍数是否数对、报告是否能生成。
"""

from __future__ import annotations

import math

from .features import FrameFeat, PlayerFeat

DEFAULT_PLAN = [
    # (活球时长秒, 拍间隔秒)
    (6.0, 1.30),
    (12.0, 1.05),
    (4.0, 1.60),
    (3.0, 2.40),
    (9.0, 1.20),
    (18.0, 0.95),
]


def _player(x, y, speed, energy=0.1, pose=False, crouch=1.8, stance=1.1,
            racket_up=True, wrist_v=0.0, accel=0.0):
    feat = PlayerFeat()
    feat.present = True
    feat.x, feat.y = x, y
    feat.px = feat.py = float("nan")
    feat.speed = speed
    feat.accel = accel
    feat.energy = energy
    feat.energy_hp = max(0.0, energy - 0.05)
    feat.h_ref = 300.0
    feat.pose_ok = pose
    feat.crouch = crouch if pose else float("nan")
    feat.stance = stance if pose else float("nan")
    feat.racket_up = racket_up if pose else None
    feat.wrist_v = wrist_v
    feat.comps = {
        "accel": abs(accel),
        "energy_hp": max(0.0, energy - 0.05),
        "wrist_v": wrist_v,
        "height_rate": 0.0,
    }
    return feat


def synth_frames(fps=10.0, plan=None, with_pose=True):
    """返回 (frames, ground_truth_rallies, hit_truth)。"""
    plan = plan or DEFAULT_PLAN
    frames = []
    truth = []
    hit_truth = []
    t = 0.0
    idx = 0

    def push(near, far, dt=1.0 / fps):
        nonlocal t, idx
        frames.append(FrameFeat(t, idx, dt, near, far))
        t += dt
        idx += 1

    for live_dur, hit_gap in plan:
        # ---- 发球准备：双方站在发球区静止 1.4s ----
        for _ in range(int(1.4 * fps)):
            push(_player(0.52, 0.45, 0.01, energy=0.05, pose=with_pose),
                 _player(0.48, 1.75, 0.01, energy=0.05))

        # ---- 活球：来回移动，交替打挥拍尖峰 ----
        start = t
        n_live = int(live_dur * fps)
        next_hit = 0.35
        side = 0
        n_hits = 0
        for k in range(n_live):
            lt = k / fps
            phase = math.sin(2 * math.pi * lt / 2.2)
            near_x = 0.5 + 0.28 * phase
            near_y = 0.55 + 0.25 * math.cos(2 * math.pi * lt / 2.2)
            far_x = 0.5 - 0.25 * phase
            far_y = 1.65 + 0.28 * math.cos(2 * math.pi * lt / 2.6)

            spike = 0.0
            if lt >= next_hit:
                spike = 1.0
                next_hit += hit_gap
                n_hits += 1
                hit_truth.append({"t": round(t, 2),
                                  "side": "near" if side == 0 else "far"})
                side ^= 1

            near_spike = spike if side != 0 else 0.0   # side 已翻转，spike 属于上一拍
            far_spike = spike - near_spike

            near = _player(near_x, near_y, 0.55 + 0.15 * abs(phase),
                           energy=0.12 + 0.35 * near_spike,
                           pose=with_pose,
                           crouch=1.55 + 0.25 * near_spike,
                           stance=1.05 + 0.2 * near_spike,
                           racket_up=near_spike > 0.5,
                           wrist_v=4.0 * near_spike,
                           accel=1.8 * near_spike)
            far = _player(far_x, far_y, 0.50 + 0.10 * abs(phase),
                          energy=0.12 + 0.35 * far_spike,
                          accel=1.6 * far_spike)
            push(near, far)

        # ---- 死球：停止 1.8s，然后走动 1.2s ----
        for _ in range(int(1.8 * fps)):
            push(_player(0.52, 0.60, 0.02, energy=0.04, pose=with_pose),
                 _player(0.48, 1.70, 0.02, energy=0.04))
        for _ in range(int(1.2 * fps)):
            push(_player(0.52, 0.25, 0.16, energy=0.05, pose=with_pose),
                 _player(0.48, 1.35, 0.14, energy=0.05))

        truth.append({"t0": round(start, 2), "t1": round(start + live_dur, 2),
                      "shots": n_hits + 1})

    return frames, truth, hit_truth
