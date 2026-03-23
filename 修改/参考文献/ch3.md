# 面向低成本无人机的非法低轨卫星终端定位识别文献深调研聚焦第三章

## 范围界定与筛选规则

本次调研的主目标是为你论文第三章“复杂城市环境下，基于 RSSI、AoA、PDR 等多源观测，利用强化学习（尤其 DQN 类）进行自适应融合定位”补齐**可直接落笔引用**的高质量文献，并兼顾第一章/第二章/第四章可复用的研究脉络段落（每个方向 150–250 字）。本回答优先选取近五年（约 2020–2026）在 entity["organization","IEEE","professional society for ee"]、entity["company","Elsevier","academic publisher"]、entity["company","Springer","academic publisher"]、entity["organization","ACM","computing society"] 等渠道可检索的论文，同时保留少量奠基文献（如 Q-learning、DQN 等）作为方法学“出处型引用”。citeturn5view0turn17search0turn19search0turn17search5turn17search2

你明确要求排除的低相关方向，本调研在选文时做了如下约束：  
不以“普通 WiFi 室内定位/指纹”作为主体文献池；不收“纯视觉目标检测”与“与定位观测无关的路径规划”；不收“仅做卫星通信系统参数优化、但不涉及非合作感知/定位/排查”的论文。对于“室内”场景，仅在其**方法机制能直接迁移到城市复杂传播与无人机监听场景**（例如 NLOS 识别、观测可信度估计、动态协方差/权重调节、UWB/IMU/PDR 融合范式）时少量纳入，并在条目中清晰标注“迁移方式”。citeturn22search11turn23search13turn21search21turn24search0

---

## 第三章强相关核心文献（15–25 篇）可直接引用式注释

下列 [R1]–[R20] 为第三章最相关的 20 篇（满足你“15–25 篇”要求）。每篇均给出：题名、作者、年份、来源、DOI/链接、主要贡献、与你论文第三章的对应关系、推荐引用位置、以及“更像哪类文献”。

### [R1] 多传感器融合定位从解析到学习的高质量综述（方法总纲）
**标题**：Multi-sensor integrated navigation/positioning systems using data fusion: From analytics-based to learning-based approaches  
**作者**：entity["people","Yuan Zhuang","gnss indoor positioning"] et al.  
**年份**：2023  
**来源**：entity["organization","Information Fusion","elsevier journal"]  
**DOI/链接**：10.1016/j.inffus.2023.01.025 citeturn5view0  
**主要贡献**：系统梳理多源融合定位/导航的“解析建模（KF/图优化等）—学习方法（含强化学习）”演进，并强调状态选取、可观性、时间同步等工程要点。citeturn5view0  
**与第三章相关性**：可作为你第三章“多源融合定位研究现状”的总括引文，用于把 RL 自适应加权自然嵌入融合定位发展主线。citeturn5view0  
**更适合引用位置**：第一章研究现状（融合定位综述）、第二章理论基础（融合架构/滤波范式）、第三章相关工作（“解析 vs 学习”对比动机）。  
**类型判断**：传统基线（综述/框架性）。

### [R2] 多源自适应融合的“对照组”思路（VB/鲁棒性、异常观测剔除）
**标题**：An Adaptive Multi-sensor Fusion for Intelligent Vehicle Localization  
**作者**：entity["people","Hao Zhu","vehicle localization"] et al.  
**年份**：2024  
**来源**：entity["organization","IEEE Sensors Journal","ieee journal"]  
**DOI/链接**：10.1109/JSEN.2024.3360083 citeturn8search9turn8search3  
**主要贡献**：用变分贝叶斯思路为每类传感器观测引入“是否为异常值/离群值”的隐变量，从而实现“只融合有效观测”的鲁棒定位。citeturn8search3turn8search9  
**与第三章相关性**：你的 PA-DQN 若要做“动态权重”，需要一个强对照：**非 RL 的自适应鲁棒加权/剔除**。这篇可作为“传统自适应融合”的代表，支撑你：RL 并非唯一途径，但可能更灵活。citeturn8search3turn8search9  
**更适合引用位置**：第三章相关工作（自适应/鲁棒融合）、第三章方法动机（“离群观测→权重应降低/置零”的建模启发）。  
**类型判断**：直接相关方法（鲁棒自适应融合基线）。

### [R3] 复杂 NLOS 环境下的自适应 IMU/UWB 融合（“观测质量会变”这一事实的证据）
**标题**：An Adaptive IMU/UWB Fusion Method for NLOS Indoor Positioning and Navigation  
**作者**：entity["people","Daquan Feng","imu uwb fusion"] et al.  
**年份**：2023  
**来源**：entity["organization","IEEE Internet of Things Journal","ieee journal"]  
**DOI/链接**：10.1109/JIOT.2023.3245144 citeturn22search11turn22search0  
**主要贡献**：面向 NLOS 条件下 UWB 量测可靠性骤降问题，提出自适应融合框架，将 IMU 与 UWB 在复杂遮挡/反射条件下的贡献动态化处理。citeturn22search11  
**与第三章相关性**：虽然载波与场景不同（UWB/室内），但“量测可信度在遮挡/反射切换时非平稳”与城市环境一致，可作为你第三章提出“动态权重/协方差调节”的关键证据链。citeturn22search11  
**更适合引用位置**：第三章相关工作（复杂环境下自适应融合）、第三章问题建模（观测噪声/偏置时变）。  
**类型判断**：直接相关方法（自适应融合，非 RL）。

### [R4] UWB+PDR 的 NLOS 识别与 KF 融合（与你 RSSI+PDR 融合最接近的“结构模板”）
**标题**：NLOS Identification Based UWB and PDR Hybrid Positioning System  
**作者**：entity["people","Dong-Hyun Kim","uwb pdr positioning"]; entity["people","Jong-Yun Pyun","wireless positioning"]  
**年份**：2021  
**来源**：entity["organization","IEEE Access","ieee journal"]  
**DOI/链接**：10.1109/ACCESS.2021.3098416 citeturn23search16turn23search0  
**主要贡献**：提出先做 LOS/NLOS 环境识别，再在 KF 框架内融合 UWB 与 PDR，从而在弱/强 NLOS 下提升鲁棒性。citeturn23search0turn23search16  
**与第三章相关性**：这篇非常适合作为你“RL-IFA（或 PA-DQN）之前的启发式/规则式自适应融合基线”：先判别观测质量，再调节融合策略。你可把它类比为“规则 gating”，再引出“用 DQN 学 gating/权重”。citeturn23search0turn23search16  
**更适合引用位置**：第三章相关工作（PDR 与无线融合）、第三章方法动机（“NLOS→降低无线观测可信度”）。  
**类型判断**：传统基线（经典滤波 + 质量判别）。

### [R5] 紧耦合 UWB 测距与 IMU/PDR 的 EKF 融合（EKF 结构与误差传播“出处型”）
**标题**：Tightly Coupling Fusion of UWB Ranging and IMU Pedestrian Dead Reckoning for Indoor Localization  
**作者**：entity["people","Ali Rashid","uwb imu fusion"] et al.  
**年份**：2021  
**来源**：IEEE Access（同刊不再重复实体标注）  
**DOI/链接**：10.1109/ACCESS.2021.3132645 citeturn23search13turn23search1  
**主要贡献**：在 EKF 框架下紧耦合 UWB 测距与 IMU/PDR，突出“惯导先准确后漂移；无线观测环境相关但不随时间漂移”的互补机制，并给出实验验证。citeturn23search1turn23search13  
**与第三章相关性**：你第三章的 RSSI+AoA+PDR（或 UAV 惯导）融合，可直接复用其“预测—更新—误差状态/观测方程”的组织方式，作为 EKF/ESKF 结构模板与对比基线。citeturn23search1  
**更适合引用位置**：第二章理论基础（EKF/紧耦合融合）、第三章基线算法（EKF 融合）。  
**类型判断**：传统基线（EKF 紧耦合融合）。

### [R6] 用“信道统计量”做 NLOS 识别（直接支撑你 PA 特征：延迟扩展/峭度等）
**标题**：NLOS Identification and Weighted Least-Squares Localization for UWB Systems Using Multipath Channel Statistics  
**作者**：entity["people","Ismail Guvenc","uwb localization"] et al.  
**年份**：2008  
**来源**：entity["organization","EURASIP Journal on Advances in Signal Processing","springer journal"]  
**DOI/链接**：链接：Springer 页面（含 RMS delay spread、kurtosis 等统计量）citeturn21search21  
**主要贡献**：提出基于多径统计特征（如峭度、均值超时延、RMS 时延扩展等）进行 NLOS 识别，并结合加权最小二乘实现 NLOS 缓解。citeturn21search21  
**与第三章相关性**：这篇是你 PA-DQN“物理感知状态设计”的关键来源：把“复杂传播→可观测的信道统计量→可用于权重/协方差调节”的链条写实。citeturn21search21  
**更适合引用位置**：第二章理论基础（多径/NLOS 指标）、第三章方法动机（物理特征→权重调度）。  
**类型判断**：传统基线（NLOS 识别/加权思想奠基）。

### [R7] 角度信息与 CIR 特征的 NLOS/LOS 识别（把 AoA 质量变化纳入“状态”）
**标题**：Angular Information-based NLOS/LOS Identification for Localization in Wireless Networks  
**作者**：entity["people","Cheng Huang","wireless localization"] et al.  
**年份**：2019  
**来源**：论文 PDF（公开稿）  
**DOI/链接**：链接（PDF）citeturn21search26  
**主要贡献**：利用角度信息与 CIR 派生特征（文中讨论 RMS-delay spread 等）进行 LOS/NLOS 判别，强调 NLOS 下特征统计规律变化。citeturn21search26  
**与第三章相关性**：你第三章要融合 AoA，必须论证“角度观测也会在多径/NLOS 下退化并可被检测”。该文适合支撑你对 AoA 质量指标/物理特征入状态的合理性。citeturn21search26  
**更适合引用位置**：第二章理论基础（AoA/NLOS 判别）、第三章方法动机（AoA 质量→动态权重）。  
**类型判断**：传统基线（观测质量判别）。

### [R8] 多径对宽带 TOA 定位影响的权威综述（城市 NLOS/时延扩展“理论背景”最强支撑之一）
**标题**：A Survey on the Impact of Multipath on Wideband Time-of-Arrival Based Localization  
**作者**：entity["people","Sundar Aditya","wireless localization"] et al.  
**年份**：2018  
**来源**：entity["organization","IEEE Signal Processing Magazine","ieee magazine"]  
**DOI/链接**：10.1109/MSP.2018.2818159 citeturn21search2  
**主要贡献**：系统讨论多径既可能“有害”（偏置、误检首径）也可能“可用”（先验统计/可分辨多径结构），并总结典型 NLOS 缓解思路。citeturn21search2  
**与第三章相关性**：即便你第三章主观测是 RSSI/AoA，该综述仍可作为“复杂传播导致观测失真/偏置/方差增大”的权威背景，从而引出“需要动态调权/鲁棒融合”。citeturn21search2  
**更适合引用位置**：第一章研究现状（复杂传播影响定位）、第二章理论基础（多径效应与指标）、第三章问题分析。  
**类型判断**：传统基线（奠基综述）。

### [R9] RSS+AoA 的非合作 RF 目标定位（与你“非合作式无源定位”叙事高度契合）
**标题**：Localization of a moving non-cooperative RF target in NLOS environment using RSS and AOA measurements  
**作者**：entity["people","Chi Cheng","rf localization"] et al.  
**年份**：2015  
**来源**：entity["organization","ICASSP","ieee conference"]  
**DOI/链接**：10.1109/ICASSP.2015.7178638 citeturn11search4  
**主要贡献**：在 NLOS 场景下结合 RSS 与 AoA 对移动的非合作 RF 目标进行定位建模与求解，体现“单一观测不稳→组合观测更有辨识度”。citeturn11search4  
**与第三章相关性**：这是你第三章“RSSI+AoA 融合”的高相关代表文献，可用来写“相关工作”中对 RSS/AoA 互补性与 NLOS 处理的综述，并作为 EKF 或加权融合的对照起点。citeturn11search4  
**更适合引用位置**：第三章相关工作（RSS/AoA 融合定位）、第三章基线（非 RL 的融合定位）。  
**类型判断**：直接相关方法（RSS+AoA 定位，非合作/NLOS）。

### [R10] NLOS 环境下的分布式非合作 RF 定位（把“鲁棒/分布式”写进研究现状）
**标题**：Distributed Localization of a RF Target in NLOS Environments  
**作者**：entity["people","Wenjie Xu","distributed localization"] et al.  
**年份**：2015  
**来源**：entity["organization","IEEE Journal on Selected Areas in Communications","ieee journal"]  
**DOI/链接**：10.1109/JSAC.2015.2430152 citeturn11search1turn11academia39  
**主要贡献**：面向“多数传感器无 LOS”的 NLOS 条件，提出分布式 EM 框架，融合（如 TDOA/AoA）并强调无需集中化的鲁棒定位。citeturn11academia39turn11search1  
**与第三章相关性**：虽然观测形态与场景不同，但非常适合支撑你第三章中的“复杂环境→需要鲁棒估计/分布式潜力”，并为后续多无人机协同（论文第四章/第五章）埋下伏笔。citeturn11search1turn11academia39  
**更适合引用位置**：第一章研究现状（非合作/NLOS RF 定位）、第三章相关工作（复杂环境鲁棒方法动机）。  
**类型判断**：直接相关方法（非合作/NLOS 定位代表）。

### [R11] 一种“鲁棒 EKF + 轨迹质量”融合框架（经典鲁棒滤波基线，可与 RL 调权对比）
**标题**：A Fusion Localization Method based on a Robust Extended Kalman Filter and Track Quality Method  
**作者**：entity["people","Yong Wang","robust ekf localization"] et al.  
**年份**：2019  
**来源**：entity["organization","Sensors","mdpi journal"]  
**DOI/链接**：10.3390/s19173638 citeturn19search22  
**主要贡献**：在 NLOS/误差环境中引入鲁棒 EKF，并结合轨迹质量评估抑制异常观测对状态估计的冲击。citeturn19search22  
**与第三章相关性**：这是非常合适的“非 RL 鲁棒融合”基线：你可以把“轨迹质量/一致性指标”视作你 PA-DQN 状态的一部分，进而把“启发式调权”升级为“DQN 学调权”。citeturn19search22  
**更适合引用位置**：第三章基线方法（鲁棒 EKF）、第三章方法动机（质量指标→权重调整）。  
**类型判断**：传统基线（鲁棒滤波/质量门控）。

### [R12] DDPG 用于自适应 KF 参数（把“调协方差”形式化为 MDP 的代表）
**标题**：Adaptive Kalman Filter Navigation Algorithm Based on Reinforcement Learning for Ground Vehicles  
**作者**：entity["people","Zheng Gao","reinforcement learning navigation"] et al.  
**年份**：2020  
**来源**：entity["organization","Remote Sensing","mdpi journal"]  
**DOI/链接**：链接（MDPI：Remote Sensing 2020, 12(11):1704）citeturn2view2  
**主要贡献**：将 KF 的过程噪声协方差估计问题表述为强化学习问题，采用 DDPG 学习自适应协方差更新策略，用于导航定位精度提升。citeturn0search5turn2view2  
**与第三章相关性**：你第三章的“动态权重”可以等价为“动态测量噪声/协方差缩放”，该文是极好的“RL-调参滤波”类基线出处，用于对比你的 DQN 调权是否带来更强稳健性。citeturn0search5turn2view2  
**更适合引用位置**：第三章相关工作（RL用于滤波调参）、第三章方法动机/对比实验（RL-调协方差基线）。  
**类型判断**：可借鉴的强化学习方法（RL 用于融合参数自适应，非 DQN）。

### [R13] 用 RL 做 EKF 协方差调谐（更贴近“工程实现”的会议论文）
**标题**：A Reinforcement Learning Approach for Adaptive Covariance Tuning in the Kalman Filter  
**作者**：entity["people","Jiajun Gu","kalman filter tuning"] et al.  
**年份**：2022  
**来源**：entity["organization","IMCEC","ieee conference"]  
**DOI/链接**：10.1109/IMCEC55388.2022.10020019 citeturn4view0  
**主要贡献**：将噪声协方差调谐建模为 MDP，提出 RL-aided 的协方差调节方法并落到误差状态 EKF 框架中验证改进。citeturn4view0  
**与第三章相关性**：可作为你“RL-IFA / RL 调权类基线”的工程近邻参考，尤其适合写在“与 EKF 兼容的 RL 设计”这一段，说明你并非把 RL 直接替代滤波，而是用于**调节融合超参数**。citeturn4view0  
**更适合引用位置**：第三章相关工作（RL 辅助滤波）、第三章方法设计（MDP 定义的可迁移模板）。  
**类型判断**：可借鉴的强化学习方法（RL 调协方差/权重）。

### [R14] 最新一类“RL 驱动协方差控制”的期刊工作（近五年强对照）
**标题**：Reinforcement learning-driven adaptive covariance control for robust automated INS/UWB navigation  
**作者**：entity["people","Kun Li","ins uwb navigation"] et al.  
**年份**：2026  
**来源**：entity["organization","ISA Transactions","elsevier journal"]  
**DOI/链接**：10.1016/j.isatra.2026.03.033 citeturn15search2  
**主要贡献**：面向 INS/UWB 自动化导航，利用 RL 驱动协方差自适应控制以增强鲁棒性，强调在复杂环境下观测噪声统计难以固定建模。citeturn15search2  
**与第三章相关性**：非常适合支撑你第三章“为什么要用 RL 做动态权重/协方差调节”的论证：复杂环境下噪声非平稳，固定权重不可持续。citeturn15search2  
**更适合引用位置**：第三章方法动机（RL 的必要性）、第三章对比基线（RL-协方差控制）。  
**类型判断**：可借鉴的强化学习方法（RL 调参/鲁棒融合）。

### [R15] UAV 场景下“几何自适应 DQN”实现 RF 发射源定位（强调多径城市环境）
**标题**：Geometry Adaptive Deep Q-Network for UAV-Based Emitter Localization in Cluttered RF Environments  
**作者**：entity["people","Christopher Peters","uav emitter localization"] et al.  
**年份**：2026  
**来源**：IEEE 会议论文（PDF 版号显示 IEEE）  
**DOI/链接**：链接（PDF）citeturn7view0  
**主要贡献**：提出面向密集多径环境的“几何自适应 DQN”，用于协同 UAV 通过观测（如 TDOA）驱动队形重构，以降低定位不确定度，并在仿真与数字孪生环境验证其在多径/衰落下的鲁棒性。citeturn7view0  
**与第三章相关性**：尽管该文的 RL 作用点偏“机动几何优化”，而你第三章偏“融合权重”，但它强力支撑两件事：①城市多径环境中 UAV 定位 RF 发射源是现实研究方向；②DQN 可与经典估计（如 EKF）结合并在退化环境中获得收益。citeturn7view0  
**更适合引用位置**：第一章研究现状（UAV 辅助 RF 定位 + RL）、第三章方法动机（DQN 在退化环境可用）、第四章/第五章（多无人机协同）。  
**类型判断**：可借鉴的强化学习方法（DQN 在 RF 定位中的工程范式）。

### [R16] Q-learning 奠基论文（RL-IFA 等“Q 学习基线”出处）
**标题**：Q-learning  
**作者**：entity["people","Christopher J. C. H. Watkins","q-learning author"]; entity["people","Peter Dayan","computational neuroscientist"]  
**年份**：1992  
**来源**：entity["organization","Machine Learning","springer journal"]  
**DOI/链接**：10.1007/BF00992698 citeturn19search0  
**主要贡献**：给出 Q-learning 的经典形式与收敛性质，是你在第三章设置“Q-learning/RL-IFA”类基线时最标准的出处引用。citeturn19search0  
**与第三章相关性**：用于“RL 基线”合法化：你若引入 RL-IFA（基于离散动作调权/调协方差），可直接引用该文作为 Q 学习理论来源。citeturn19search0  
**更适合引用位置**：第二章理论基础（强化学习基础）、第三章基线算法（Q-learning 调权）。  
**类型判断**：传统基线（奠基理论）。

### [R17] DQN 奠基论文（DQN/PA-DQN 最核心“出处”）
**标题**：Human-level control through deep reinforcement learning  
**作者**：entity["people","Volodymyr Mnih","deep reinforcement learning"] et al.  
**年份**：2015  
**来源**：entity["organization","Nature","scientific journal"]  
**DOI/链接**：10.1038/nature14236 citeturn17search0  
**主要贡献**：提出 DQN（经验回放、目标网络等关键机制），是你第三章提出 DQN 类方法进行动态权重学习的最核心方法学出处。citeturn17search0  
**与第三章相关性**：可用于你“PA-DQN 网络结构与训练机制”一节：为什么需要 replay buffer、target network 来稳定学习。citeturn17search0  
**更适合引用位置**：第二章理论基础（DQN）、第三章方法（DQN 架构与训练）。  
**类型判断**：可借鉴的强化学习方法（方法奠基）。

### [R18] Double DQN（解决 Q 值过估计，支撑你“稳健性改进”设计）
**标题**：Deep Reinforcement Learning with Double Q-Learning  
**作者**：entity["people","Hado van Hasselt","reinforcement learning"] et al.  
**年份**：2016  
**来源**：entity["organization","AAAI Conference on Artificial Intelligence","ai conference"]  
**DOI/链接**：10.1609/aaai.v30i1.10295 citeturn17search5  
**主要贡献**：指出 DQN 存在 Q 值过估计问题，并提出 Double Q-learning 的深度版本（Double DQN）以缓解过估计、提升学习稳定性。citeturn17search5  
**与第三章相关性**：若你的 PA-DQN 需要强调“动态权重在复杂环境中不应被极端奖励误导”，可引用 Double DQN 作为稳定性增强的标准改造；也适合作为你对比实验的算法变体。citeturn17search5  
**更适合引用位置**：第三章方法（DQN 稳定化改进）、第三章消融实验（DQN vs Double DQN）。  
**类型判断**：可借鉴的强化学习方法（DQN 变体）。

### [R19] Dueling DQN（把“状态价值/动作优势”拆分，适合做“权重动作空间”）
**标题**：Dueling Network Architectures for Deep Reinforcement Learning  
**作者**：entity["people","Ziyu Wang","deep reinforcement learning"] et al.  
**年份**：2016  
**来源**：entity["organization","PMLR","ml proceedings publisher"]（ICML 2016 论文集页面）  
**DOI/链接**：链接（PMLR 论文页/开放稿）citeturn17search2turn17search6  
**主要贡献**：提出 Dueling 架构，把表征拆为 V(s) 与 A(s,a)，在“动作之间差异较小/相近”时可改善学习效率与策略评估。citeturn17search2turn17search6  
**与第三章相关性**：你的动态权重动作往往是“有限档位/相近调整”（例如 {降低、保持、提高} 或多档缩放），非常符合 Dueling DQN 的适用直觉，可作为 PA-DQN 的结构性增强引用。citeturn17search2turn17search6  
**更适合引用位置**：第三章方法（Dueling 结构动机）、第三章消融（普通 DQN vs Dueling DQN）。  
**类型判断**：可借鉴的强化学习方法（DQN 变体）。

### [R20] 用 DQN 做“传感器选择/调度”的统一建模（对“动态权重=动态选择”的强类比）
**标题**：Deep Reinforcement Learning Sensor Scheduling for Effective Monitoring of Dynamical Systems  
**作者**：M. Alali; entity["people","Mahdi Imani","sensor scheduling"]（文中作者信息）  
**年份**：2024  
**来源**：PMC 收录全文（期刊信息以原文为准）  
**DOI/链接**：链接（PMC）citeturn16view0  
**主要贡献**：将“从多个传感器子集里选择/调度”形式化为基于后验信念状态的 RL 问题，并指出 DQN 适合“连续状态（belief）+离散动作（传感器子集）”的结构。citeturn16view0  
**与第三章相关性**：你可以把“动态权重”解释为“软选择”（soft selection）：在每个时刻选择更可信的一组观测贡献更大。该文对你第三章 MDP 要素定义（状态=可信度/统计量，动作=权重档位，奖励=定位误差下降/一致性提升）有直接“范式迁移”价值。citeturn16view0  
**更适合引用位置**：第三章方法动机（为何用 RL）、第三章 MDP 建模（状态/动作/奖励设计模板）。  
**类型判断**：可借鉴的强化学习方法（DQN 用于动态选择/调度）。

---

## 跨章节补充锚点文献（用于“方向综述段落”与第一章/第四章铺垫）

以下 [R21]–[R27] 不要求你第三章逐篇深用，但对“机会信号（Starlink/LEO）”“Radio Map/REM”“知识蒸馏鲁棒协同”三条主线的综述段落非常关键，且能把你的论文题目（非法 LEO 终端排查定位）从“泛定位”拉回到“LEO 机会信号与非合作感知”的主问题上。

### [R21] 低成本接收条件下利用 Starlink 下行纯音做机会定位
**标题**：Practical Use of Starlink Downlink Tones for Positioning  
**作者**：N. Jardak et al.  
**年份**：2023  
**来源**：PMC 收录全文  
**DOI/链接**：链接（PMC）citeturn24search0  
**要点**：验证无需大型抛物面天线、使用低成本 LNB 等即可跟踪 Starlink 下行纯音并用于机会定位。citeturn24search0

### [R22] 盲接收解析 Starlink OFDM-like 参考信号结构并实现米级定位
**标题**：Unveiling Starlink LEO Satellite OFDM-Like Signal Structure Enabling Precise Positioning  
**作者**：S. Kozhaya / A. Kassas 团队论文（公开 PDF）  
**年份**：2024  
**来源**：公开 PDF（Kassas 团队发布）  
**DOI/链接**：链接（PDF）citeturn24search5  
**要点**：揭示 Starlink 下行 OFDM-like 参考信号结构并提出盲接收机，实验显示可同时跟踪多颗卫星并获得米级水平误差。citeturn24search5

### [R23] “海量卫星机会信号”PNT 的概念框架与研究路线
**标题**：Toward Massive Satellite Signals of Opportunity Positioning  
**作者**：G. Fan et al.  
**年份**：2024  
**来源**：Science 系列 SPJ：Space  
**DOI/链接**：10.34133/space.0191 citeturn24search31  
**要点**：系统讨论利用大量非 GNSS 卫星下行信号进行 PNT 的总体思路与挑战，为你第一章“LEO 机会信号定位”写研究现状提供高层框架。citeturn24search31

### [R24] REM/Radio Map 的图学习重构（稀疏监测数据→完整电磁场图）
**标题**：Reconstruction of Radio Environment Map Based on Multi-Complete Graph Structure  
**作者**：X. Wen et al.  
**年份**：2024  
**来源**：Sensors（MDPI）  
**DOI/链接**：10.3390/s24082523 citeturn24search21  
**要点**：提出用图神经网络在缺少显式传播图结构条件下，从稀疏频谱监测数据推断并补全 REM。citeturn24search21

### [R25] KD 近年进展综述（掉点鲁棒协同、轻量化部署的“方法学总引文”）
**标题**：A survey on knowledge distillation: Recent advancements  
**作者**：A. Moslemi et al.  
**年份**：2024  
**来源**：ScienceDirect（期刊综述）  
**DOI/链接**：链接（ScienceDirect）citeturn24search3  
**要点**：总结知识蒸馏在结构、训练范式与应用域的近期发展，适合你第四章/第五章写“Teacher–Student”“轻量化鲁棒协同”的总引文。citeturn24search3

### [R26] KD + 自适应协方差 EKF 的定位融合范式（极贴合你“掉点+蒸馏鲁棒”）
**标题**：KD-EKF: Knowledge-Distilled Adaptive Covariance EKF for Robust UWB/PDR Indoor Localization  
**作者**：K. Yoo et al.  
**年份**：2026  
**来源**：arXiv 预印本  
**DOI/链接**：链接（arXiv）citeturn15academia15  
**要点**：将“UWB/PDR 融合”明确改写为“动态可信度/不确定度估计”问题，采用 teacher–student 蒸馏得到轻量模型在线调节 EKF 协方差，专门针对 LOS/NLOS 切换下误差尖峰。citeturn15academia15

### [R27] 物理先验注入的稀疏 Radio-Map 重构（与“PA-DQN/物理感知”一脉相承）
**标题**：Physics-Informed Representation Alignment for Sparse Radio-Map Reconstruction  
**作者**：H. Jia  
**年份**：2025  
**来源**：ACM（论文页）  
**DOI/链接**：链接（ACM DL）citeturn24search25  
**要点**：把物理信息/先验通过表示对齐注入稀疏无线电地图重构，适合你将“物理感知”扩展到 Radio Map/REM 分支。citeturn24search25

---

## 可直接粘贴进论文的“方向综述段落”与挂引用建议

下列每段约 150–250 字，语气按硕士论文常见写法组织，并给出“最适合挂的引用编号”（对应本回答的 [R•]）。

**低轨卫星机会信号与非合作感知**（适合第一章“国内外研究现状”）  
近年来，低轨宽带星座下行信号被用于“机会定位/机会感知”，研究重点从纯音跟踪逐步扩展到对未知 OFDM-like 参考信号的盲解析与多星联合观测，强调在低成本接收链路（如通用 LNB、非定制天线）下仍可提取稳定可用的多普勒/载波等观测量。然而，由于信号体制非公开、星地相对运动快且城市遮挡显著，观测的可用性与质量高度时变，如何把“可解析的信号结构/观测量”与“复杂传播下的鲁棒估计”耦合仍是难点。citeturn24search0turn24search5turn24search31  
推荐引用：[R21], [R22], [R23]

**复杂城市传播机理与多径/NLOS对定位的影响**（适合第二章“理论基础”与第三章“问题分析”）  
城市峡谷环境下的遮挡、反射与散射会导致 RSS、AoA 乃至时延观测出现偏置与方差膨胀，且在 LOS/NLOS 切换时呈现突变，单一观测源往往难以长期稳定工作。已有研究一方面从多径统计角度总结了时延扩展、峭度等指标与 NLOS 的相关性，另一方面也指出多径既可能破坏首径检测，也可在具备先验或结构可分辨时被利用。对融合定位而言，更关键的是把这些“传播退化的可观测证据”转化为在线权重/协方差调节依据，从而抑制异常观测对估计的主导效应。citeturn21search21turn21search26turn21search2turn19search22  
推荐引用：[R6], [R7], [R8], [R11]

**UAV-assisted localization与机动感知**（适合第一章与第四章过渡）  
无人机因具备机动部署与高度优势，常被用作移动传感平台对地面无线发射源进行定位与监测。在复杂多径环境中，传统基于固定几何或贪心准则的机动策略易陷入次优，而将机动/队形调整或观测策略设计建模为强化学习问题，可在奖励函数中同时考虑不确定度收敛、信号强度与航迹代价，从而在退化传播下仍获得更快的定位收敛。该思路为你论文从“单机融合定位”走向“多机协同/空间采样与策略学习”提供了合理研究脉络。citeturn7view0turn11search1turn8search0  
推荐引用：[R15], [R10]

**RSSI/AoA/PDR/EKF等经典方法与融合范式**（适合第二章“理论基础”、第三章“基线方法”）  
经典定位通常围绕量测模型与状态估计展开：RSS/AoA 等无线观测提供绝对/相对约束，PDR/惯导提供连续运动先验但存在漂移，二者在 EKF 等贝叶斯滤波框架下形成“预测—更新”的融合结构。现实复杂环境中，关键瓶颈不在滤波框架本身，而在于观测噪声与偏置的统计难以固定建模；因此常见改进包括 NLOS 识别、鲁棒估计与引入轨迹质量/一致性指标来抑制异常观测。你第三章的贡献点可以被清晰表述为：在经典滤波骨架上，用学习策略替代手工调参，实现在线自适应融合。citeturn23search0turn23search13turn19search22turn11search4  
推荐引用：[R5], [R4], [R11], [R9]

**多源异构融合的自适应加权与鲁棒融合**（适合第三章“相关工作”）  
多源融合定位的核心是利用互补性抵消单源缺陷，但在城市多径与遮挡条件下，观测质量高度非平稳，使得固定权重或固定噪声协方差往往在特定场景失效。近期工作中，一类方法通过为观测引入“是否为离群/可靠”的隐变量，或用贝叶斯推断让系统自动选择有效量测，从而实现鲁棒融合；另一类则在紧耦合融合中强调误差传播机理与观测互补的结构性优势。你可据此把“自适应加权”具体化为“动态可信度估计→权重/协方差调节→误差尖峰抑制”。citeturn8search3turn8search9turn22search11turn23search1  
推荐引用：[R2], [R3], [R5]

**强化学习用于定位/传感器融合/动态权重**（适合第三章“方法动机”与“MDP定义”）  
将融合权重或滤波参数在线调节视为序贯决策问题，可自然映射为 MDP：状态刻画当前观测质量与一致性特征，动作对应权重/协方差缩放策略，奖励与定位误差下降、创新一致性或不确定度收敛相关。相比启发式调参，RL 方法不依赖固定噪声先验，能在多场景数据驱动下学习到“何时信谁、信多少”的策略；其中 DQN 及其稳定化变体（如 Double/Dueling）为离散动作的权重档位调整提供了成熟实现路径。citeturn2view2turn4view0turn15search2turn17search0turn17search5turn17search2  
推荐引用：[R12], [R13], [R14], [R17], [R18], [R19], [R20]

**Radio Map/REM重构与多目标定位识别**（适合第四章“多目标定位与识别”）  
在多目标或信号源密集场景中，逐目标估计易受数据关联与遮挡影响，因而“先重构空间信号场/Radio Map，再进行定位识别”的两阶段范式更具可扩展性。REM 重构的关键在于稀疏采样条件下对空间相关性与传播规律的建模：一方面可利用图学习在缺少显式图结构时完成从稀疏监测到全局补全，另一方面也可注入物理先验提升跨场景泛化能力。你可将 UAV 机动采样视为“主动获取最有信息的采样点”，与第三章的“动态调权”在方法论上保持一致。citeturn24search21turn24search25turn24search10  
推荐引用：[R24], [R27]

**知识蒸馏与掉点鲁棒协同定位**（适合第四章/第五章“多节点协同与掉点鲁棒”）  
多无人机协同定位面临通信受限与节点掉点，模型需要在分布式条件下保持鲁棒且具备轻量化部署能力。知识蒸馏通过 teacher–student 结构把复杂模型的表征与决策能力迁移到轻量学生模型，有助于在算力与带宽受限时维持稳定表现；更进一步的研究将蒸馏与滤波融合结合，使学生模型在线输出测量可信度或协方差缩放，用于抑制 LOS/NLOS 切换带来的误差尖峰，从而在异构环境下减少人工调参依赖。该方向可直接服务你第四章“掉点条件下的鲁棒协同定位”。citeturn24search3turn15academia15  
推荐引用：[R25], [R26]

---

## 按章节归位建议（哪些文献放哪章最划算）

第一章“国内外研究现状”最适合用来搭骨架的引用组合：用 [R23] 交代“LEO 机会信号 PNT”总体路线；用 [R21]–[R22] 说明 Starlink 信号从纯音到 OFDM-like 可解析观测的发展；用 [R10]、[R15] 体现“非合作 RF 目标 + UAV 平台 + 复杂环境鲁棒性”的研究主线；用 [R1] 给出“融合定位从解析到学习”的大趋势。citeturn24search31turn24search0turn24search5turn11search1turn7view0turn5view0

第二章“理论基础”可形成三段支撑链：  
其一，用 [R8] 讲多径对定位观测的根本影响，再落到 [R6]、[R7] 的“可观测物理特征（时延扩展/角度信息等）→NLOS 识别”；其二，用 [R5]、[R11] 交代 EKF/鲁棒 EKF 的融合骨架与质量门控；其三，用 [R16]–[R19] 奠基 RL/DQN 关键概念与稳定化技巧。citeturn21search2turn21search21turn21search26turn23search13turn19search22turn19search0turn17search0turn17search5turn17search2

第三章“单目标自适应融合定位”建议把“对照组”写齐：  
非 RL 自适应鲁棒融合：[R2], [R3], [R4], [R11]；  
非合作 RSS/AoA 定位代表：[R9], [R10]；  
RL 调参/调权最近邻基线：[R12], [R13], [R14]；  
DQN 与变体“方法出处”：[R17], [R18], [R19]（必要时加 [R16] 作为 Q-learning 基线）。citeturn8search9turn22search11turn23search0turn19search22turn11search4turn11search1turn2view2turn4view0turn15search2turn17search0turn17search5turn17search2turn19search0

第四章“多目标定位与知识蒸馏”可用 [R24], [R27] 支撑 Radio Map/REM 的“稀疏补全 + 物理先验”；用 [R25]–[R26] 支撑 teacher–student 蒸馏与“在线可信度/协方差调节”，并用 [R10], [R15] 把多机协同与复杂环境鲁棒性串到你的主问题上。citeturn24search21turn24search25turn24search3turn15academia15turn11search1turn7view0

---

## 优先级推荐清单与“第三章写作落点”提示

### 必引文献（第三章最能“撑住论证链”）
[R1]（综述定调：解析→学习融合定位主线）citeturn5view0  
[R6]+[R7]（物理特征/NLOS 识别：支撑“PA”状态设计）citeturn21search21turn21search26  
[R9]（RSS+AoA 的非合作/NLOS 目标定位：支撑“RSSI+AoA 融合必要性”）citeturn11search4  
[R11]（鲁棒 EKF + 质量门控：强传统基线）citeturn19search22  
[R12] 或 [R14]（RL 驱动协方差/权重自适应：支撑“为什么要 RL”）citeturn2view2turn15search2  
[R17]+[R18]+[R19]（DQN/Double/Dueling 出处：支撑 PA-DQN 设计合理性与消融）citeturn17search0turn17search5turn17search2

### 可选补充文献（视你第三章实验设计取舍）
[R2]（VB 异常观测建模：与你“动态权重”做高质量对照）citeturn8search9turn8search3  
[R3]、[R5]（IMU/UWB/PDR 紧耦合范式：补充“观测互补机制”的结构证据）citeturn22search11turn23search1  
[R13]（会议型 RL 协方差调谐：强调工程可落地）citeturn4view0  
[R20]（DQN 做动态选择/调度的范式：帮助你把 MDP 三要素写得更“像论文”）citeturn16view0

### 不建议在第三章大篇幅引用但可了解的边缘文献
[R15]（更偏“机动几何策略学习”，与第三章“融合调权”需明确区分作用点，适合放在第一章/后续章节）citeturn7view0  
[R24]/[R27]（REM/Radio Map 更偏第四章主线，第三章仅需点到“可拓展到多目标”即可）citeturn24search21turn24search25  
[R25]/[R26]（KD 更适合第四章/第五章，第三章若写会分散主线）citeturn24search3turn15academia15

---

## 额外点名：最适合支撑 PA-DQN 与基线出处的文献组合

**最适合支撑“PA-DQN 设计合理性”的组合（建议第三章方法动机/状态设计处成组引用）**  
用 [R6]、[R7] 把“时延扩展/角度信息等物理统计量可指示 NLOS/多径退化”落地；用 [R11] 说明传统鲁棒融合依赖质量指标/门控；再用 [R17]–[R19] 给出 DQN 及稳定化变体的标准出处，形成“物理可观测证据 → 需要动态调权 → 用 DQN 学策略”的闭环。citeturn21search21turn21search26turn19search22turn17search0turn17search5turn17search2

**最适合作为 EKF、RL-IFA 等基线出处的文献**  
EKF/鲁棒滤波基线：以 [R5]（紧耦合 EKF 融合范式）+ [R11]（鲁棒 EKF/质量门控）作为“结构出处 + 鲁棒化出处”，在论文写作上通常比引用极早期 KF 推导更贴合“定位融合”语境。citeturn23search1turn19search22  
RL-IFA（Q-learning 调权/调参）基线出处：用 [R16] 奠基 Q-learning；若你的 RL-IFA 采用“状态=质量指标，动作=权重档位，奖励=误差下降”这种标准定义，可再并列引用 [R20] 作为“把选择/调度写成 MDP”的近年范式参考。citeturn19search0turn16view0