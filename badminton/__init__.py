"""badminton-analyzer: 离线羽毛球视频回合切分与近端姿态分析。

模块划分（自底向上）：
    cvreq       OpenCV 懒加载与友好报错
    config      全部可调参数
    court       球场单应变换（像素 <-> 球场单位）
    videoio     视频探测、按目标帧率抽帧、元数据
    detect      球员检测与跟踪（运动前景 / 可插拔 YOLO）
    pose        姿态后端（MediaPipe / 代理模式 / 关闭）
    features    逐帧特征与平滑
    rally       回合状态机
    hits        挥拍事件候选 + 交替约束动态规划解码
    metrics     回合指标与全局聚合
    clips       回放片段的裁切渲染
    viz         叠加层绘制
    report      Markdown 报告生成
    pipeline    编排
"""

__version__ = "0.1.0"
