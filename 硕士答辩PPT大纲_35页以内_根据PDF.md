# 硕士毕业答辩 PPT 大纲（35 页以内，依据 PDF 完整论文）

论文题目：基于无人机载平台的卫星互联网终端高精度定位方法

依据 PDF：`/Users/miles/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_q7h4p62hbgta22_1198/temp/RWTemp/2026-05/9f8a1def339b79a75552a38c3bed5cca/2023140064-石志鹏-基于无人机平台的非法终端定位算法.pdf`

校验结果：该 PDF 与工程文件 `BUPTGraduateThesisLatexTemplate/out/2023140064-石志鹏-基于无人机平台的非法终端定位算法.pdf` 完全一致，MD5 为 `26d2277d45f9fb79203d0cae9c9c7de1`，共 111 页。

说明：下文中的“PDF 页”指 PDF 阅读器显示的页码，不是论文正文页脚页码。大纲严格按 PDF 目录和正文组织：第 1 章绪论、第 2 章系统模型与理论基础、第 3 章 PA-DQN 单目标自适应融合定位、第 4 章 Radio Map 与知识蒸馏多目标协同鲁棒定位、第 5 章结论与展望。

## 一、答辩整体主线

建议将答辩压缩为 34 页，控制在 18-22 分钟。主线如下：

1. 任务对象：异常线索触发后的非合作低轨卫星互联网终端定位排查，不替代协议认证或法律裁定。
2. 平台选择：低成本无人机具有机动性和视角优势，但受载荷、续航、算力和观测条件限制。
3. 核心困难：复杂城市环境下存在多径、遮挡、非视距、观测质量动态波动；多目标场景还有稀疏扫描、局部多峰、多径鬼影和无人机节点掉点。
4. 方法一：对单目标连续定位，提出基于物理感知深度强化学习的 PA-DQN 自适应融合定位方法。
5. 方法二：对多目标协同定位，提出基于 Radio Map 空间信号表征和教师—学生知识蒸馏的鲁棒定位方法。
6. 结论：传播机理建模、空间/状态表征、智能学习与鲁棒机制结合，可以提升复杂城市环境下低成本无人机定位精度和工程鲁棒性。

## 二、34 页 PPT 逐页大纲

| 页码 | PPT 标题 | 核心内容 | 直接引用图片/表格 | 讲述重点 |
|---:|---|---|---|---|
| 1 | 封面 | 论文题目、姓名、学号、导师、学院、专业、答辩日期 | 无 | 使用 PDF 封面题目：基于无人机载平台的卫星互联网终端高精度定位方法 |
| 2 | 汇报提纲 | 研究背景与问题、系统模型、单目标 PA-DQN 方法、多目标 Radio Map + KD 方法、总结展望 | 无 | 让评委先看到全文结构，避免后面细节分散 |
| 3 | 研究背景：低轨卫星互联网终端监管需求 | LEO 卫星终端规模增长，异常接入、异常辐射和疑似违规使用带来无线电安全监管问题 | 可不用图，或使用简洁流程示意 | 强调论文任务是“异常线索后的定位排查”，不是直接判定非法性 |
| 4 | 任务边界与无人机平台优势 | 监管系统/运营商给出粗粒度可疑区域，低成本 UAV 进入区域进行机动观测和定位 | 图 2-1，PDF 页 25；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch2/1_无人机机动观测下非法终端定位场景示意图.jpg` | 讲清楚“为什么需要无人机”：机动、灵活、视角可调，但平台资源受限 |
| 5 | 复杂城市环境下的关键挑战 | 非合作信号稳定性不足；单目标观测质量动态波动；多目标稀疏扫描、局部多峰、鬼影和掉点并存 | 可自绘三栏问题图 | 这页对应 PDF 第 22 页的三个主要问题，是后面两章方法的出发点 |
| 6 | 论文研究内容与技术路线 | 系统建模与传播机理分析；单目标自适应融合定位；多目标协同鲁棒定位 | 图 1-1，PDF 页 23；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch1/全文结构图.png` | 用“单目标 -> 多目标 -> 残缺观测鲁棒性”的递进关系串起全文 |
| 7 | 系统任务场景与统一观测模型 | 目标状态集合、无人机观测集合、状态方程和观测方程；观测映射受传播环境与噪声影响 | 可引用图 2-1，或自绘状态-观测框图 | 不展开复杂公式，强调“定位误差来自目标状态、传播环境、观测方式共同作用” |
| 8 | 传播机理与经典定位方法局限 | 多径、时延扩展、阴影衰落和 NLOS 使 RSSI/AOA/TOA/TDOA/PDR 都存在适用边界 | 可选四宫格：图 2-2 到图 2-5，PDF 页 30-32；原图见 `figures/ch2/AOA.jpg`、`TOA.jpg`、`TDOA.jpg`、`PDR.jpg` | 只讲和后续有关的结论：单一观测量不稳定，所以需要融合和智能决策 |
| 9 | 智能决策与知识迁移基础 | MDP/DQN 支撑动态调权，知识蒸馏支撑残缺观测下的能力迁移 | 图 2-6，PDF 页 34；图 2-7，PDF 页 37；原图：`figures/ch2/MDP.jpg`、`figures/ch2/KD.jpg` | 作为方法铺垫，控制在 1 分钟内 |
| 10 | 单目标连续定位问题 | RSSI 易受遮挡和衰落影响，AOA 易受强反射影响，PDR 短时平滑但长期漂移 | 图 3-1，PDF 页 42；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/1_单目标非法终端定位场景示意图.png` | 引出固定权重和静态融合规则在复杂环境下不可靠 |
| 11 | 单目标 PA-DQN 总体框架 | 多源观测模块、物理感知特征提取、PA-DQN 决策、自适应加权融合、奖励反馈闭环 | 图 3-2，PDF 页 43；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/系统架构图.png` | 这是第 3 章最重要的框架图，建议单独一页放大 |
| 12 | 多源加权融合模型 | RSSI、AOA、PDR 分别输出初步位置估计，通过动态权重融合得到最终位置 | 图 3-5，PDF 页 49；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/5_定位输出链与状态构造链并行关系图.png` | 讲清楚本文不是直接融合原始信号，而是在位置输出链和状态构造链上并行处理 |
| 13 | 调权问题的 MDP 建模 | 状态表示环境与观测质量，动作表示权重更新，奖励评价定位误差与轨迹平滑性 | 图 3-3，PDF 页 46；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/2_单目标自适应融合定位MDP建模示意图.png` | 强调权重调整是序列决策，不是单步静态优化 |
| 14 | 物理感知状态空间 | 状态向量由 RMS 时延扩展、RSSI 方差、RF-PDR 几何一致性、上一时刻权重构成 | 图 3-6，PDF 页 50；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/6_物理感知状态空间构成示意图.jpg` | 这是方法创新核心：让智能体在误差放大前感知环境退化 |
| 15 | PA-DQN 状态与融合决策关系 | PA-DQN 不再被动等待后验误差，而是根据物理复杂度和多源一致性前置调权 | 图 3-9，PDF 页 58；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/图39物理感知状态.png` | 与上一页可二选一；如果时间紧，保留第 14 页、删第 15 页 |
| 16 | 离散增量动作与奖励设计 | 动作采用细粒度权重增量，奖励由定位误差和平滑惩罚组成，抑制权重突变和轨迹锯齿 | 图 3-10，PDF 页 59；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/10_PADQN机制.jpg` | 讲清楚“增量动作 + 平滑奖励”主要改善连续轨迹稳定性 |
| 17 | PA-DQN 网络与训练流程 | 物理感知状态输入，DQN 输出候选动作 Q 值；经验回放和目标网络稳定训练 | 图 3-11，PDF 页 61；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/DQN网络架构图.png` | 不要讲伪代码细节，只说明输入、输出和训练机制 |
| 18 | 单目标实验设置 | 基于 OSM 数据和射线追踪构建复杂城市场景；轨迹包括矩形、对角线、反对角线、随机游走；总测试里程超过 200 m | 图 3-4，PDF 页 48；图 3-12，PDF 页 64；原图：`figures/ch3/3_复杂城市三维场景建模示意图.jpg`、`figures/ch3/11_PADQN训练收敛过程示意图.png` | 说明实验可控、覆盖 LOS/NLOS 切换和多类轨迹 |
| 19 | 单目标总体性能对比 | PA-DQN 在总体测试集上 RMSE = 2.1468 m，P95 = 3.8724 m，优于 PDR-only、AOA-only、EKF、RL-IFA | 表 3-4，PDF 页 65 | 建议把表 3-4 重绘为柱状图；保留数值，突出 PA-DQN 相较 EKF/RL-IFA 的优势 |
| 20 | 不同轨迹场景性能 | 线性、矩形、随机游走场景中 PA-DQN 均取得最低 RMSE 和 P95 | 表 3-5 到表 3-7，PDF 页 65-66 | 说明方法不依赖单一轨迹，随机游走最难但仍稳定 |
| 21 | 典型轨迹可视化 | PA-DQN 在矩形拐角、对角线穿越、随机游走中更贴近真实轨迹，局部抖动更弱 | 图 3-13，PDF 页 68-69；原图：`figures/ch3/12a_矩形轨迹对比图.png`、`12b_对角线轨迹对比图.png`、`12c_反对角线轨迹对比图.png`、`12d_随机游走轨迹对比图.png` | 可以拆成 2×2 排版，少放文字 |
| 22 | 信噪比鲁棒性与消融实验 | 低 SNR 下 PA-DQN 的 RMSE/P95 增长更平缓；完整 PA-DQN RMSE 2.15、权重变化率 0.04，去除物理感知状态后 RMSE 升至 4.89 | 图 3-14、图 3-15，PDF 页 70；表 3-8，PDF 页 71；原图：`figures/ch3/13_不同算法多种信噪比RMSE对比曲线.png`、`figures/ch3/14_不同算法多种信噪比P95对比曲线.png` | 把鲁棒性和消融合并，节省页数；结论是物理感知状态贡献精度，增量动作/稳定奖励贡献平滑性 |
| 23 | 单目标方法小结 | PA-DQN 实现从“后验误差修正”到“物理先验感知驱动调权”的转变，提升精度、尾部误差和连续轨迹稳定性 | 可用三点总结 | 作为第 3 章收束，过渡到多目标场景 |
| 24 | 多目标并发定位问题 | 多目标场景不仅要估计位置，还要面对目标数量不确定、采样稀疏、局部多峰和多径鬼影 | 图 4-1，PDF 页 73；图 4-2，PDF 页 74；原图：`figures/ch4/第四章图.png`、`figures/ch4/1_多目标并发定位场景示意图.jpg` | 讲清楚第 4 章相对第 3 章的扩展：由单目标轨迹估计到区域级空间判别 |
| 25 | 稀疏采样与局部多峰问题 | UAV 空间扫描只能获得有限离散观测，噪声和采样不均匀会使单一目标附近分裂出多个局部峰值 | 图 4-3，PDF 页 77；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/图片.png` | 说明传统峰值检测会把局部伪峰误判为目标 |
| 26 | Radio Map 信号成像思路 | 将离散 RSSI 样本映射为二维无线电热力图，把“点峰值检测”转化为“区域级判别” | 图 4-5，PDF 页 80；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/4_离散观测到RadioMap成像流程图.jpg` | 这是第 4 章第一条核心方法线 |
| 27 | OSM + MATLAB 工具链与传播场建模 | 基于 OSM、MATLAB Site Viewer、Antenna Toolbox、RF Propagation Toolbox 建模，获得覆盖目标区域 27,546 个网格点的传播数据 | 图 4-4，PDF 页 79；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/7_OSM与MATLAB工具链建模流程图.png` | 强调 Radio Map 有物理建模依据，不是简单图像插值 |
| 28 | Radio Map 三维可视化 | 城市峡谷、反射增强、阴影衰落等条件下，空间功率分布具有明显非均匀峰谷结构和区域纹理 | 图 4-6、图 4-7，PDF 页 82-83；原图：`figures/ch4/11_城市峡谷RadioMap三维图A.png`、`12_城市峡谷RadioMap三维图B.png`、`13_反射增强RadioMap三维图C.png`、`14_阴影衰落RadioMap三维图D.png` | 这页适合放大图片，展示复杂传播场的区域结构 |
| 29 | 多无人机掉点与残缺观测 | 16 架 UAV 协同观测时，节点掉点会导致空间覆盖、几何基线和 Radio Map 结构同时退化 | 可自绘 16 节点掉点示意，或引用图 4-1 的右侧流程 | 引出为什么需要知识蒸馏，而不是只靠插值补全 |
| 30 | 教师—学生知识蒸馏建模 | 教师网络利用完整观测学习全局空间判别分布，学生网络在残缺观测下通过硬标签和软标签联合训练逼近教师能力 | 可用公式 `L = alpha L_ce + beta L_kd` | 强调蒸馏迁移的是完整网络中的全局空间判别知识，不是一般意义上的模型压缩 |
| 31 | 教师—学生 GAT 网络架构 | 节点特征为 `[x_i, y_i, z_i, R_i]`，全连接图建模，多层 GAT 提取节点关系，全局平均池化后输出 400 个栅格类别 | 图 4-8，PDF 页 89；原图：`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/KD网络架构图.png` | 这是第 4 章第二条核心方法线，建议重点讲 |
| 32 | 多目标实验设置与复杂度 | 区域 500 m × 500 m，20 × 20 栅格，400 类；16 架 UAV；掉点比例 0%、12.5%、25%、37.5%、50%；Student-KD 测试复杂度与 Student-NoKD 相同 | 表 4-1、表 4-2，PDF 页 93-94 | 讲实验公平性和工程代价：蒸馏主要增加训练开销，推理阶段不增加学生模型复杂度 |
| 33 | 25% 掉点条件下性能对比 | Student-KD：Mean Error 12.8 m、RMSE 16.0 m、90% Error 26.4 m；Student-NoKD：22.4/27.9/46.5；Student-KD 约 40% 误差下降并接近 Teacher | 图 4-9、图 4-10，PDF 页 96-97；表 4-3，PDF 页 96；原图：`figures/ch4/9_不同模型平均定位误差对比图.png`、`figures/ch4/10_不同模型九十分位误差对比图.jpg` | 这是第四章最重要实验页，数字要讲清楚 |
| 34 | 总结与展望 | 总结：PA-DQN 提升单目标连续定位精度与鲁棒性；Radio Map + KD 提升多目标残缺观测鲁棒定位能力。展望：真实场景验证、三维定位、多目标数量估计、动态拓扑和更丰富蒸馏机制 | 表 4-4 可放入备份页；PDF 页 98 | 最后一页明确贡献与不足，不建议再加入新技术内容 |

## 三、重要图片直接引用清单

做 PPT 时优先插入下面这些工程原图，不建议从 PDF 截图。原图路径均为绝对路径，直接在 PowerPoint 中选择“插入 -> 图片 -> 此设备”，粘贴或定位到对应路径即可。图注建议保留论文图号，格式为：

`图 X-X  原论文图题（引自本人硕士论文，PDF 阅读器第 X 页）`

| 论文图号 | PDF 页 | PPT 建议用途 | 原图绝对路径 | PPT 图注建议 |
|---|---:|---|---|---|
| 图 1-1 | 23 | 全文技术路线 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch1/全文结构图.png` | 图 1-1 全文结构图（引自本人硕士论文，PDF 阅读器第 23 页） |
| 图 2-1 | 25 | 任务场景与 UAV 平台优势 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch2/1_无人机机动观测下非法终端定位场景示意图.jpg` | 图 2-1 无人机机动观测下的非合作终端定位场景（引自本人硕士论文，PDF 阅读器第 25 页） |
| 图 3-1 | 42 | 单目标定位场景 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/1_单目标非法终端定位场景示意图.png` | 图 3-1 复杂城市环境下单目标非合作终端定位场景（引自本人硕士论文，PDF 阅读器第 42 页） |
| 图 3-2 | 43 | PA-DQN 总体框架，必须重点引用 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/系统架构图.png` | 图 3-2 基于物理感知深度强化学习的单目标自适应融合定位总体框架图（引自本人硕士论文，PDF 阅读器第 43 页） |
| 图 3-3 | 46 | MDP 建模 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/2_单目标自适应融合定位MDP建模示意图.png` | 图 3-3 单目标自适应融合定位问题的马尔可夫决策过程建模（引自本人硕士论文，PDF 阅读器第 46 页） |
| 图 3-4 | 48 | 单目标实验环境 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/3_复杂城市三维场景建模示意图.jpg` | 图 3-4 基于 OSM 数据的复杂城市三维场景建模（引自本人硕士论文，PDF 阅读器第 48 页） |
| 图 3-5 | 49 | 定位输出链与状态构造链 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/5_定位输出链与状态构造链并行关系图.png` | 图 3-5 定位输出链与状态构造链的并行结构关系（引自本人硕士论文，PDF 阅读器第 49 页） |
| 图 3-6 | 50 | 物理感知状态空间，必须重点引用 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/6_物理感知状态空间构成示意图.jpg` | 图 3-6 物理感知状态空间构成（引自本人硕士论文，PDF 阅读器第 50 页） |
| 图 3-9 | 58 | PA-DQN 状态与融合决策关系 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/图39物理感知状态.png` | 图 3-9 PA-DQN 物理感知状态空间及其与融合决策关系（引自本人硕士论文，PDF 阅读器第 58 页） |
| 图 3-10 | 59 | 离散增量动作与权重更新 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/10_PADQN机制.jpg` | 图 3-10 PA-DQN 离散增量动作空间与权重更新机制（引自本人硕士论文，PDF 阅读器第 59 页） |
| 图 3-11 | 61 | PA-DQN 网络结构 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/DQN网络架构图.png` | 图 3-11 PA-DQN 中 Q 网络结构（引自本人硕士论文，PDF 阅读器第 61 页） |
| 图 3-12 | 64 | 训练收敛过程 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/11_PADQN训练收敛过程示意图.png` | 图 3-12 PA-DQN 训练阶段的收敛过程（引自本人硕士论文，PDF 阅读器第 64 页） |
| 图 3-13 | 68-69 | 典型轨迹对比，必须引用 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/12a_矩形轨迹对比图.png`；`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/12b_对角线轨迹对比图.png`；`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/12c_反对角线轨迹对比图.png`；`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/12d_随机游走轨迹对比图.png` | 图 3-13 不同算法在典型测试轨迹下的连续定位结果对比（引自本人硕士论文，PDF 阅读器第 68-69 页） |
| 图 3-14 | 70 | SNR-RMSE 鲁棒性 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/13_不同算法多种信噪比RMSE对比曲线.png` | 图 3-14 不同算法在多种信噪比条件下的 RMSE 对比曲线（引自本人硕士论文，PDF 阅读器第 70 页） |
| 图 3-15 | 70 | SNR-P95 鲁棒性 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/14_不同算法多种信噪比P95对比曲线.png` | 图 3-15 不同算法在多种信噪比条件下的 P95 对比曲线（引自本人硕士论文，PDF 阅读器第 70 页） |
| 图 4-1 | 73 | 第四章总体思路 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/第四章图.png` | 图 4-1 第四章多目标协同定位方法总体思路（引自本人硕士论文，PDF 阅读器第 73 页） |
| 图 4-2 | 74 | 多目标并发定位场景 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/1_多目标并发定位场景示意图.jpg` | 图 4-2 基于低成本无人机平台的地面多目标非合作终端并发定位场景（引自本人硕士论文，PDF 阅读器第 74 页） |
| 图 4-3 | 77 | 稀疏采样与局部多峰 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/图片.png` | 图 4-3 无人机稀疏采样与噪声扰动条件下的局部多峰现象（引自本人硕士论文，PDF 阅读器第 77 页） |
| 图 4-4 | 79 | OSM 与 MATLAB 工具链 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/7_OSM与MATLAB工具链建模流程图.png` | 图 4-4 基于 OSM 与 MATLAB 工具链的复杂城市场景建模流程（引自本人硕士论文，PDF 阅读器第 79 页） |
| 图 4-5 | 80 | Radio Map 成像流程，必须重点引用 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/4_离散观测到RadioMap成像流程图.jpg` | 图 4-5 从无人机离散观测到 Radio Map 成像及跨模态区域判别的整体流程（引自本人硕士论文，PDF 阅读器第 80 页） |
| 图 4-6 | 82 | Radio Map 三维可视化 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/11_城市峡谷RadioMap三维图A.png`；`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/12_城市峡谷RadioMap三维图B.png` | 图 4-6 复杂城市传播场条件下 Radio Map 的三维可视化结果（一）（引自本人硕士论文，PDF 阅读器第 82 页） |
| 图 4-7 | 83 | Radio Map 三维可视化 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/13_反射增强RadioMap三维图C.png`；`/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/14_阴影衰落RadioMap三维图D.png` | 图 4-7 复杂城市传播场条件下 Radio Map 的三维可视化结果（二）（引自本人硕士论文，PDF 阅读器第 83 页） |
| 图 4-8 | 89 | 教师—学生网络架构，必须重点引用 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/KD网络架构图.png` | 图 4-8 教师—学生网络架构（引自本人硕士论文，PDF 阅读器第 89 页） |
| 图 4-9 | 96 | 25% 掉点误差 CDF | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/9_不同模型平均定位误差对比图.png` | 图 4-9 不同方法在 25% 掉点条件下的定位误差累计分布曲线（引自本人硕士论文，PDF 阅读器第 96 页） |
| 图 4-10 | 97 | 25% 掉点误差柱状图 | `/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch4/10_不同模型九十分位误差对比图.jpg` | 图 4-10 不同方法在 25% 掉点条件下的误差统计柱状图（引自本人硕士论文，PDF 阅读器第 97 页） |

## 四、表格引用建议

论文中的表格建议不要截图，最好在 PPT 中重绘为柱状图或简化表格。引用格式同样保留表号：

| 表号 | PDF 页 | 建议用途 | 核心数值 |
|---|---:|---|---|
| 表 3-4 | 65 | 单目标总体性能对比 | PA-DQN：RMSE 2.1468 m，P95 3.8724 m；RL-IFA：RMSE 5.9002 m，P95 10.5578 m；EKF：RMSE 7.3364 m，P95 13.0217 m |
| 表 3-5 到表 3-7 | 65-66 | 不同轨迹场景性能 | PA-DQN 在矩形、线性、随机游走三类场景均最低：RMSE 分别为 2.2846 m、1.8765 m、2.5496 m |
| 表 3-8 | 71 | PA-DQN 消融实验 | 完整 PA-DQN：RMSE 2.15 m、权重变化率 0.04；去除物理感知状态：RMSE 4.89 m；去除增量式动作：权重变化率 0.28 |
| 表 4-3 | 96 | 25% 掉点性能对比 | Student-KD：Mean 12.8 m、RMSE 16.0 m、90% Error 26.4 m；Student-NoKD：22.4/27.9/46.5 |
| 表 4-4 | 98 | 不同掉点比例鲁棒性 | 50% 掉点时 Student-NoKD Mean Error 38.9 m，Student-KD 17.3 m |

## 五、答辩时建议重点讲的 8 页

如果答辩时间紧，重点讲以下页面：

1. 第 5 页：复杂城市环境下的关键挑战。
2. 第 6 页：论文研究内容与技术路线。
3. 第 11 页：单目标 PA-DQN 总体框架。
4. 第 14 页：物理感知状态空间。
5. 第 19 页：单目标总体性能对比。
6. 第 26 页：Radio Map 信号成像思路。
7. 第 31 页：教师—学生 GAT 网络架构。
8. 第 33 页：25% 掉点条件下性能对比。

## 六、图片引用的实际操作方式

### PowerPoint 中直接插入原图

1. 打开 PPT 对应页。
2. 选择“插入 -> 图片 -> 此设备”。
3. 复制上表中的原图绝对路径，定位到文件后插入。
4. 图下方保留论文图号和题名，例如：
   `图 3-2 基于物理感知深度强化学习的单目标自适应融合定位总体框架图（引自本人硕士论文，PDF 阅读器第 43 页）`

### 如果使用 Markdown/Marp/Typora 做初稿

可直接用 Markdown 图片语法引用，例如：

```markdown
![图 3-2 基于物理感知深度强化学习的单目标自适应融合定位总体框架图](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/系统架构图.png)
```

### 如果使用 LaTeX Beamer

在 `.tex` 中写：

```tex
\begin{figure}
  \centering
  \includegraphics[width=0.92\textwidth]{/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/figures/ch3/系统架构图.png}
  \caption{图 3-2 基于物理感知深度强化学习的单目标自适应融合定位总体框架图}
\end{figure}
```

注意：答辩 PPT 里引用的是你自己的论文图片，不需要写外部文献引用，但建议保留“图号 + 图题 + PDF 页码”，这样评委追问时可以迅速对应到论文正文。
