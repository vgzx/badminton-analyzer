"""回合切分状态机（不依赖球轨迹）。

判据
----
1. **发球准备**：双方都站在各自发球区、速度低于 ``still_speed`` 且持续
   ``serve_still_duration`` -> 这是最硬的回合起点锚点，不需要球。
2. **进入活球**：任意一方速度超过 ``move_on_speed``（发球挥拍 / 接发启动）。
3. **待定区 QUIET**：双方同时静止 ``quiet_duration``。这一态的存在是为了
   不被“高远球对拉时两个人站着等”误切成两个回合。
4. **确认死球**：从 QUIET 里出现**持续走动**（> ``walk_on_speed`` 维持
   ``walk_duration``）——捡球、走回发球位。这才是真正的死球证据。

因此“静止”本身永远不是回合结束的判据，只是进入待定的条件。
"""

from __future__ import annotations

from .config import FAR_SERVICE_Y, NEAR_SERVICE_Y, SERVICE_X


IDLE, SERVE, LIVE, QUIET = "idle", "serve", "live", "quiet"


class RallySegmenter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = IDLE
        self.rallies = []
        self._id = 0
        self.t_serve = 0.0
        self.t_live = 0.0
        self.t_quiet = 0.0
        self.last_active = 0.0
        self.still_t = 0.0
        self.quiet_t = 0.0
        self.walk_t = 0.0
        self.missing_t = 0.0
        self.cold_t = 0.0
        self.last_t = 0.0

    # ------------------------------------------------------------------
    def _in_service(self, feat, near: bool) -> bool:
        if not feat.present:
            return False
        pad = self.cfg.serve_zone_pad
        lo, hi = NEAR_SERVICE_Y if near else FAR_SERVICE_Y
        if not (lo - pad <= feat.y <= hi + pad):
            return False
        return SERVICE_X[0] - pad <= feat.x <= SERVICE_X[1] + pad

    # ------------------------------------------------------------------
    def update(self, t, feat):
        """喂入一帧特征，返回本帧新产生的回合（通常是空列表）。"""
        cfg = self.cfg
        dt = max(feat.dt, 1e-3)
        self.last_t = t
        near, far = feat.near, feat.far
        both = near.present and far.present
        speeds = [p.speed for p in (near, far) if p.present]
        max_speed = max(speeds) if speeds else 0.0

        new = []
        if not both:
            self.missing_t += dt
            if self.missing_t > 3.0 and self.state == SERVE:
                self._reset_idle()
        else:
            self.missing_t = 0.0

        # ---------------- IDLE ----------------
        if self.state == IDLE:
            if both and self._in_service(near, True) and self._in_service(far, False) \
                    and max_speed < cfg.still_speed:
                self.still_t += dt
                if self.still_t >= cfg.serve_still_duration:
                    self.state = SERVE
                    self.t_serve = t
                    self.walk_t = 0.0
            else:
                self.still_t = 0.0
                # 冷启动：视频从中途开始、直接就是活球
                if cfg.allow_cold_start and max_speed > cfg.move_on_speed:
                    self.cold_t += dt
                    if self.cold_t >= 1.0:
                        self.state = LIVE
                        self.t_live = max(0.0, t - self.cold_t - 0.3)
                        self.last_active = t
                        self.quiet_t = 0.0
                        self.cold_t = 0.0
                else:
                    self.cold_t = 0.0

        # ---------------- SERVE ----------------
        elif self.state == SERVE:
            if max_speed > cfg.move_on_speed:
                self.state = LIVE
                self.t_live = max(self.t_serve, t - 0.25)
                self.last_active = t
                self.quiet_t = 0.0
            elif t - self.t_serve > 12.0:
                self._reset_idle()
            elif both and not (self._in_service(near, True) or self._in_service(far, False)):
                self._reset_idle()

        # ---------------- LIVE ----------------
        elif self.state == LIVE:
            self.last_active = t
            if both and max_speed < cfg.still_speed:
                self.quiet_t += dt
            else:
                self.quiet_t = 0.0
            if self.quiet_t >= cfg.quiet_duration:
                self.state = QUIET
                self.t_quiet = t - cfg.quiet_duration
                self.walk_t = 0.0
            elif t - self.t_live > cfg.max_rally_duration:
                new.append(self._emit(self.last_active, "超时强制结束"))

        # ---------------- QUIET ----------------
        elif self.state == QUIET:
            if max_speed > cfg.walk_on_speed:
                self.walk_t += dt
            else:
                self.walk_t = max(0.0, self.walk_t - 0.5 * dt)

            if max_speed > cfg.move_on_speed * 1.6:
                # 明显又动起来了：刚才只是高远球中的停顿，回到 LIVE
                self.state = LIVE
                self.quiet_t = 0.0
                self.walk_t = 0.0
                self.last_active = t
            elif self.walk_t >= cfg.walk_duration:
                new.append(self._emit(self.t_quiet + 0.25, "捡球/走动"))
            elif t - self.t_quiet > 15.0:
                new.append(self._emit(self.t_quiet + 0.25, "长时间静止"))

        for rally in new:
            self._reset_idle()
        return new

    # ------------------------------------------------------------------
    def _emit(self, t1, reason):
        t0 = max(0.0, self.t_live)
        t1 = max(t0 + 0.2, min(t1, self.last_t))
        dur = t1 - t0
        kind = "rally"
        if dur < self.cfg.min_rally_duration:
            kind = "serve_fault"
        elif self.t_live <= 1e-6:
            kind = "partial"
        rally = {
            "id": self._id,
            "t0": round(t0, 3),
            "t1": round(t1, 3),
            "duration": round(dur, 3),
            "kind": kind,
            "end_reason": reason,
        }
        self._id += 1
        self.rallies.append(rally)
        return rally

    def _reset_idle(self):
        self.state = IDLE
        self.still_t = 0.0
        self.quiet_t = 0.0
        self.walk_t = 0.0
        self.cold_t = 0.0

    # ------------------------------------------------------------------
    def finish(self):
        """视频结束时把未闭合的回合强制收尾。"""
        out = []
        if self.state in (LIVE, QUIET):
            out.append(self._emit(self.last_active if self.last_active else self.last_t,
                                  "视频结束"))
            self._reset_idle()
        return out
