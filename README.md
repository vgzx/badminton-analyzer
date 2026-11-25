# badminton-analyzer

离线、本地的羽毛球视频分析：**自动切分有效回合 → 统计拍数 → 分析近端球员姿态 → 导出回合片段与两份 Markdown 报告**。
不联网、不上传视频、不需要 GPU（CPU 可跑），除 OpenCV 外没有重依赖。

> 设计文档见 [docs/DESIGN.md](docs/DESIGN.md)：里面写清了每个判据、公式、参数取值理由和已知失效场景。
> 建议先读设计文档的“局限”一节，再决定是否投入。

## 1. 它产出什么

一次运行结束，你会拿到：

| 产物 | 路径 | 说明 |
|---|---|---|
| 回合片段 | `out/clips/r012_05m33s.mp4` | 每个有效回合一个 mp4，自动去掉捡球和停顿 |
| 回合描述文档 | `out/report_rallies.md` | 总览、回合列表、拍数分布、时间趋势、重点回合逐拍时间线 |
| 姿态分析文档 | `out/report_pose.md` | 蹲伏、站距、持拍手抬起率、回位时间、按拍数/按时间分组对比 |
| 回合指标表 | `out/rallies.csv` | 一行一个回合，可直接进 Excel |
| 结构化数据 | `out/metrics.json` / `out/hits.json` | 给二次开发或接自己的看板用 |
| 运行参数 | `out/run_config.json` | 复现本次结果所需的全部参数 |

## 2. 它不做什么（很重要）

本版本**不使用羽毛球轨迹**，所以：

- ❌ 没有落点、没有球路、没有“杀球/吊球/高远球”分类；
- ❌ 不知道谁赢了这一分（因此报告里不做“赢分 vs 输分”对比，只做“长回合 vs 短回合”“前半场 vs 后半场”对比）；
- ❌ 不判界内界外。

换来的好处是：**一台没有独显的家用电脑就能跑完一整场**，且状态机逻辑可解释、可调参、出错时能定位到具体判据。

拍数不是靠时长估的，靠“挥拍事件 + 交替约束”解码；详见设计文档 4.5。

## 3. 安装

```bash
python -m pip install -r requirements.txt
```

只依赖 `numpy` + `opencv-python-headless`。可选：

```bash
pip install mediapipe     # 真实姿态（不装则自动降级为「几何代理」模式）
pip install ultralytics   # YOLO 检测后端（默认不用）
```

## 4. 快速开始（三步）

### 第一步：自检（不需要视频，10 秒）

```bash
python run.py --selftest --out out_selftest
```

它会用合成数据跑完整条链路，并打印：

```
自检结果：真值 6 个回合，检出 6 个，容差内匹配 6 个（召回 100%），拍数平均误差 0.50
```

同时生成一份示例报告，可以先看看报告长什么样。**如果这一步失败，先解决环境问题，别急着跑视频。**

### 第二步：用合成视频验证视频链路（1 分钟）

```bash
python tools/make_demo_video.py --out demo.mp4
python run.py --input demo.mp4 --out out_demo
```

demo 视频里预置了 6 个回合（节奏固定），跑完对比 `out_demo/report_rallies.md` 的回合数，
可以确认解码、检测、片段导出是否正常。

### 第三步：分析真实比赛

```bash
# 建议先只跑前 3 分钟摸底
python run.py --input match.mp4 --out out_fast --max-seconds 180

# 确认没问题后跑全场
python run.py --input match.mp4 --out out
```

推荐先做球场标定（否则距离是相对单位）：

```bash
python tools/pick_court.py --input match.mp4 --at 30 --out court.json
python run.py --input match.mp4 --out out --court court.json
```

标定顺序：**近端左角 → 近端右角 → 远端左角 → 远端右角**（站在底线后面向球网）。
点完按 `S` 保存；工具会把整个球场反投影画在画面上，线必须压住真实白线，否则按 `R` 重选。

## 5. 拍摄要求

| 项目 | 要求 | 不满足会怎样 |
|---|---|---|
| 机位 | 底线后、抬高、整场入画 | 回合切分和姿态全部失真 |
| 稳定性 | 全程不动 | 标定失效，距离全错 |
| 光照 | 稳定，别逆光 | 前景检测漏人 |
| 帧率 | 25/30fps 足够，1080p 足够 | 更高帧率只增加耗时 |
| 场景 | **单打**、固定场地 | 双打需改判据（按半场聚合） |

## 6. 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `-i/--input` | — | 输入视频 |
| `-o/--out` | `out` | 输出目录 |
| `--court` | 无 | 球场四点 JSON（`tools/pick_court.py` 生成） |
| `--pose` | `auto` | `auto`/`mediapipe`/`proxy`/`off` |
| `--detector` | `motion` | `motion`（纯前景，无需模型）/`yolo` |
| `--fps` | `10` | 分析帧率，越大越慢越准 |
| `--max-seconds` | 0 | 只分析前 N 秒 |
| `--no-clips` | 关 | 不导出片段（想快速看指标时用） |
| `--overlay` | 关 | 片段上叠加脚点与球场线 |
| `--config` | 无 | 从 JSON 读参数 |
| `--save-config` | 无 | 保存当前参数后退出 |
| `--selftest` | 关 | 合成数据自检 |

全部参数（含阈值）见 `badminton/config.py`，每个都写了单位和含义。

## 7. 报告怎么读

**先看三个自检指标**，它们决定报告里的数字能不能信：

1. `有效时间占比` —— 通常 25%~40%。如果 >70%，八成是把死球算成回合了；
2. `一致性检查` 里的 `拍数与时长相关性` —— 应在 0.5~0.85。低于 0.5 说明拍数检测不稳；
3. 回合列表里的 `可信度` 列 —— 标“低”的回合不要拿去做统计。

**然后看结论**：

- `拍数分布`：长回合那一档的平均跑动是否明显升高 → 长回合是不是在耗你体能；
- `时间趋势`：最后 5 分钟的跑动/回位时间 vs 前 5 分钟 → 体能衰减曲线；
- `姿态分析` 里的 `持拍手抬起率` → 低于 40% 基本可以确定等球时拍子是放下的，这是最容易改的一项；
- `回位时间` → 击球后回不到位的直接量化。

## 8. 调参指引（按症状查）

| 症状 | 调什么 |
|---|---|
| 回合被切成很多碎段 | 提高 `--fps`；增大 `quiet_duration`（1.5→2.0）；增大 `walk_duration` |
| 两个回合被粘成一个 | 降低 `walk_on_speed`（0.10→0.08）；缩短 `quiet_duration`；检查是否有人一直走动 |
| 拍数普遍偏少 | 降低 `hit_z_threshold`（0.9→0.7）；检查 `--pose` 是否降级成了 proxy |
| 拍数普遍偏多 | 提高 `hit_z_threshold`；增大 `hit_gap_min` |
| 检出率为 0 或双人可见率 <55% | 换 `--detector yolo`；检查逆光；降低 `min_box_h_frac` |
| 片段太多太碎 | 提高 `min_rally_duration`（1.0→2.0） |
| 太慢 | 降 `--fps`（10→6）；关掉 `--overlay`；`--no-clips` 先看指标 |

参数改完请用 `--max-seconds 180` 做回归对比，别在全场视频上试错。

**Windows 控制台中文乱码**：先执行 `chcp 65001`，或设置环境变量 `PYTHONIOENCODING=utf-8`。
报告文件本身是 UTF-8 编码，乱码只影响终端显示。

## 9. 目录结构

```
badminton-analyzer/
├── run.py                   CLI 入口
├── requirements.txt
├── badminton/
│   ├── config.py            全部参数（带单位说明）
│   ├── court.py             单应变换与坐标单位（纯 numpy，可测）
│   ├── cvreq.py             OpenCV 懒加载
│   ├── videoio.py           探测、抽帧、裁剪
│   ├── detect.py            前景检测 + 近/远端指派（可换 YOLO）
│   ├── pose.py              MediaPipe / 几何代理 / 关闭
│   ├── features.py          逐帧特征与平滑
│   ├── rally.py             回合状态机
│   ├── hits.py              挥拍事件 + 交替约束 DP
│   ├── metrics.py           回合指标与聚合
│   ├── clips.py             片段导出
│   ├── viz.py               叠加层
│   ├── report.py            两份 Markdown 报告
│   ├── synthetic.py         合成特征流（自检/测试用）
│   └── pipeline.py          编排 + selftest
├── tools/
│   ├── pick_court.py        四点标定（鼠标交互）
│   └── make_demo_video.py   生成合成测试视频
├── tests/                   25 个测试，不需要 cv2
└── docs/DESIGN.md           实现方案说明
```

## 10. 验证与已知限制

**已验证**（本机无 OpenCV 环境下）：

- 25 个单元测试全部通过：标定往返精度 <1e-6 像素、状态机 6 种场景、DP 交替约束、指标聚合、报告渲染；
- `--selftest` 端到端：6/6 回合、召回 100%、拍数平均误差 0.5 拍。

**未在本机验证**：真实视频的解码/编码路径（`detect.py`、`clips.py`、`--overlay`）——因为本机没有安装 OpenCV。
这部分代码已逐行对照 OpenCV 4.x API 复核，但**请你先用第 4 节第二步的 demo 视频跑一遍**再上真实素材。

**算法层面的限制**（不可通过调参消除）：

1. 没有球 → 没有落点/球路/得分归属；
2. 单目 2D → 无转体角度、拍面朝向、击球点高度；
3. 多球训练没有“发球准备”段，状态机不会启动；
4. 双打（4 人）需要改造球员指派逻辑；
5. 长高远球对拉时双方静止，靠“待定态 + 走动判据”规避误切，极端情况仍可能少切一刀。

## 11. 许可

本目录代码可自由修改与商用（依赖库各自的许可证请自行遵守，注意 Ultralytics 是 AGPL，仅在启用 `--detector yolo` 时涉及）。
