"""全部可调参数集中在这里，便于复现与调参。

单位约定（贯穿全项目）
---------------------
* **球场单位 u**：1 u = 球场宽度 6.10 m。因此场地长 13.40 m = 2.20 u，
  单打边线 x = 0.46 / 5.64 m = 0.075 / 0.925 u，近端前发球线 y = 4.72 m = 0.774 u。
* 速度阈值统一用 u/s 表示。换算：0.11 u/s ≈ 0.67 m/s。
* 未标定时无法得到真实米制尺度，此时 u 退化为“画面宽度归一化”，
  所有距离指标一律标注为相对单位，不做米制杜撰。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
COURT_LENGTH_U = COURT_LENGTH_M / COURT_WIDTH_M  # 2.1967


@dataclass
class Config:
    # ---------- 输入 / 采样 ----------
    analysis_fps: float = 10.0      # 分析帧率（抽帧后的目标帧率）
    detect_width: int = 320         # 检测用缩略图宽度，越小越快
    max_seconds: float = 0.0        # >0 时只分析前 N 秒，便于快速摸底
    detector: str = "motion"        # motion | yolo

    # ---------- 前景检测 ----------
    bg_history: int = 300
    bg_var_threshold: int = 20
    min_area_px: int = 80           # 缩略图上的最小前景面积
    min_box_h_frac: float = 0.06    # 框高下限（占画面高）
    max_box_h_frac: float = 0.55    # 框高上限，超过视为有人从镜头前走过
    track_gate_frac: float = 0.18   # 跟踪关联的最大位移（占画面高）
    track_max_miss: int = 25        # 连续丢失多少分析帧后放弃旧轨迹

    # ---------- 回合状态机（u/s）----------
    still_speed: float = 0.11       # 视为“静止”的速度上限
    move_on_speed: float = 0.16     # 从发球准备进入 live 的速度
    walk_on_speed: float = 0.10     # 死球后“开始走动”的速度
    serve_still_duration: float = 0.8   # 双方在发球区静止多久算发球准备
    quiet_duration: float = 1.5     # 双方同时静止多久进入待定
    walk_duration: float = 0.55     # 走动持续多久确认死球
    min_rally_duration: float = 1.0  # 短于此判为发球失误（保留但标记）
    max_rally_duration: float = 90.0
    serve_zone_pad: float = 0.06    # 发球区外扩，容忍站位偏差
    allow_cold_start: bool = True   # 视频从活球中途开始时，允许直接起一个回合

    # ---------- 挥拍事件 ----------
    hit_gap_min: float = 0.35       # 两次击球最小间隔（人体反应极限）
    hit_gap_max: float = 8.0        # 最大间隔（高远球滞空）
    hit_typical_gap: float = 1.6    # 正常回合的典型击球间隔
    hit_gap_penalty: float = 0.35   # 间隔偏离典型值的惩罚系数
    hit_z_threshold: float = 0.9    # 候选点稳健 z 分数下限
    hit_smooth_sec: float = 0.12    # 候选序列平滑窗口

    # ---------- 姿态 ----------
    pose_backend: str = "auto"      # auto | mediapipe | proxy | off
    pose_every: int = 2             # 每 N 个分析帧做一次姿态
    pose_crop_scale: float = 1.35   # 以球员框为中心向外扩的比例
    wrist_smooth_sec: float = 0.10

    # ---------- 平滑 ----------
    speed_tau: float = 0.20         # 速度 EMA 时间常数（秒）
    energy_tau: float = 0.12
    baseline_tau: float = 8.0       # 动作能量的慢速基线

    # ---------- 输出 ----------
    clip_pad_before: float = 0.6    # 片段前置留白
    clip_pad_after: float = 0.9     # 片段后置留白
    clip_fourcc: str = "mp4v"
    export_clips: bool = True
    overlay_clips: bool = False     # 片段上叠加脚点/状态
    export_full_overlay: bool = False
    progress_every: int = 200

    # ---------- 标定 ----------
    court_json: str = ""            # 球场四点 JSON 路径

    def copy(self) -> "Config":
        return Config(**asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def merge(self, **kwargs) -> "Config":
        cfg = self.copy()
        for key, value in kwargs.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


# 发球区（球场单位）。近端球员在 y<=0.774u，远端在 y>=1.42u。
NEAR_SERVICE_Y = (0.0, 0.774)
FAR_SERVICE_Y = (1.42, COURT_LENGTH_U)
SERVICE_X = (0.0, 1.0)


def u_to_m(value_u: float) -> float:
    """球场单位 -> 米。"""
    return float(value_u) * COURT_WIDTH_M
