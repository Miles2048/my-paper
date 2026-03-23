# 第一章“国内外研究现状”文献骨架与可直接落笔的综述抓手

## 使用范围与写作定位

你这篇论文的“第一章研究现状”应当围绕**“非法低轨卫星终端的非合作式无源定位”**这一核心问题，回答三件事：现有研究**做到什么程度**、主流方法**怎么做**、关键难点与不足**在哪里**，并自然引出你后续章节的技术路线（单目标多源自适应融合、多目标 Radio Map/REM、多无人机/掉点与知识蒸馏鲁棒协同）。这一章的一条主线、四条分支与后续章节的一一对应关系，在你的论文材料中已有明确设定。fileciteturn0file0

下文按你指定的四条技术主线给出：发展脉络（100–200字）、4–8篇最适合写入“研究现状综述”的代表文献（每篇给出“可用于第一章哪段 + 能支撑的观点”），以及该方向的共同不足与如何承接到你的研究。

---

## 低轨卫星机会信号利用与 Starlink/LEO 下行链路解析

### 发展脉络总结（100–200字）

近五年，LEO 通信星座信号被系统性纳入“机会信号 PNT”研究：先由综述归纳场景、观测量与接收机形态，再以 Starlink 为代表开展下行信号结构逆向工程，逐步从“用纯音/载波相位做可行性验证”推进到“识别 OFDM/参考信号并实现高增益处理”，并开始严肃讨论星载/地面链路时统不连续、时钟抖动等对绝对测距的根本限制。适合挂引文：[1]–[6]。citeturn4view1turn1view0turn10view1turn6view0turn6view1turn6view2

### 推荐文献列表（4–8篇）

**[1] Survey on Opportunistic PNT With Signals From LEO Communication Satellites**  
年份：2025（早期在线/卷期信息以出版为准）  
来源：IEEE Communications Surveys & Tutorials  
核心贡献：系统梳理 LEO 通信星座作为 SoOP（signals of opportunity）用于定位/授时的**研究版图、信号体制差异、可用观测量、接收机与算法链路**，并总结主要挑战与开放问题。citeturn4view1  
可用于第一章哪一段：1.2 开头“总体研究版图与趋势”段（作为顶层综述锚点）。  
它能支撑的观点：LEO-SoOP/PNT 已从零散个例进入“可体系化总结与比较”的阶段，研究热点集中在**信号结构解析、时频同步、观测量提取与鲁棒性**。

**[2] Signal Structure of the Starlink Ku-Band Downlink**  
年份：2022/2023（以期刊版本标注）  
来源：IEEE Transactions on Aerospace and Electronic Systems  
核心贡献：给出 Starlink Ku 频段下行的**信号结构解析**（同步序列/帧结构等关键线索），为后续定位观测量提取提供“可对齐、可相关”的结构基础。citeturn1view0  
可用于第一章哪一段：1.2.1 “Starlink/LEO 下行体制解析与非合作感知”段（里程碑工作）。  
它能支撑的观点：对 Starlink 这类非合作信号，**“可定位”的前提首先是“可感知/可解析”**——结构被识别后，相关积累与参数估计才可成立。

**[3] The First Carrier Phase Tracking and Positioning Results With Starlink LEO Satellite Signals**  
年份：2022  
来源：IEEE Transactions on Aerospace and Electronic Systems（短文/通信类）  
核心贡献：在无需完全公开体制的条件下，展示对 Starlink 信号进行**载波相位跟踪并实现定位的首批结果**，证明“Starlink 可用于高精度观测量”的可行性边界。citeturn10view1  
可用于第一章哪一段：1.2.1 中“从可行性验证到结构化处理”的过渡段。  
它能支撑的观点：即便在结构信息不充分时，仍可通过频域特征（如纯音/窄带分量）获得**高精度相位类观测**，但可扩展性与鲁棒性受限。

**[4] Unveiling Starlink LEO Satellite OFDM-Like Signal Structure Enabling Precise Positioning**  
年份：2024  
来源：IEEE Transactions on Aerospace and Electronic Systems  
核心贡献：进一步揭示 Starlink 下行中**类 OFDM 参考/同步结构**，面向定位给出更稳定、可重复的结构化处理路径（更高处理增益、更可控的参数估计）。citeturn6view0  
可用于第一章哪一段：1.2.1 中“结构解析推动定位精度跃迁”的主论段。  
它能支撑的观点：从“盲跟踪”走向“参考信号驱动的结构化估计”，是 Starlink/LEO-SoOP 从演示走向工程化定位的关键台阶。

**[5] Unveiling Starlink for PNT**  
年份：2025  
来源：NAVIGATION（ION 官方期刊）  
核心贡献：在“结构已被识别”的基础上，更工程化地呈现 Starlink 用于 PNT 的信号形态、可用参考与处理链路，并讨论可达精度与实现要点。citeturn6view1  
可用于第一章哪一段：1.2.1 末尾“可用 PNT 能力与工程实现讨论”段。  
它能支撑的观点：Starlink 作为 PNT 机会信号具备现实潜力，但**依赖可持续的体制可解析性、稳定的时频特性与可获得的辅助信息**。

**[6] Timing Properties of the Starlink Ku-band Downlink and Implications for Positioning**  
年份：2025（预印本）  
来源：arXiv（建议在文中标注“预印本/待正式出版”）  
核心贡献：系统测量并讨论 Starlink 下行的**时统/定时特性（如时钟不连续、抖动等）**，指出其对伪距类绝对测距与长期稳健定位的影响与限制。citeturn6view2  
可用于第一章哪一段：1.2.1 的“现有不足/关键挑战”段（非常适合用来“点题难点”）。  
它能支撑的观点：LEO-SoOP 的核心瓶颈之一在于**时间基准不可控**，导致伪距/TOA 类观测存在系统性不确定来源，必须用鲁棒建模与多源融合对冲。

### 共同不足（面向你的选题聚焦）

现有 LEO-SoOP/Starlink 研究多以“**我方自主接收定位**”为目标，常默认可获得一定辅助信息（如卫星轨道、星历、可长期跟踪的参考结构），而对“**定位地面非法终端**”这种对抗式任务，仍缺少：面向城市遮挡场景的可探测性评估、对终端侧发射体制/间歇特性的建模、以及与 UAV 机动采样/多源融合紧耦合的系统性算法框架。citeturn4view1turn6view2

### 第一章如何承接到你的研究

建议在该小节结尾用“两句话”承接：  
其一，Starlink/LEO 下行解析工作证明了“非合作信号结构可被逆向与利用”，但对抗式终端排查仍面临**间歇发射 + 城市多径/NLOS + 低成本载荷约束**三重挑战，因此需要把“信号解析—观测提取—鲁棒融合—机动采样”作为闭环系统来设计。citeturn1view0turn6view0turn6view2  
其二，你的论文将以低成本 UAV 为平台，将目标从“自定位 PNT”转向“对地非合作终端定位识别”，并在后续章节引入自适应融合、REM 重构、多机协同与知识蒸馏来提升鲁棒性。fileciteturn0file0

---

## 复杂城市环境传播机理与经典定位技术退化问题

### 发展脉络总结（100–200字）

城市峡谷环境下，定位退化的研究已从“经验误差”走向“可建模的偏差来源”：一方面通过标准化信道模型刻画多径簇、角扩展、时延扩展与遮挡统计；另一方面在 UAV 空地链路与卫星-地面链路中引入仰角相关的 LOS 概率与路径损耗模型；与此同时，定位算法层面对 NLOS 偏差的鲁棒估计成为主线，并逐步与“把多径当信息”的新一代定位思想结合。适合挂引文：[7]–[12]。citeturn15view1turn31search10turn33search1turn34view0turn35view1turn39search0

### 推荐文献列表（4–8篇）

**[7] 3GPP TR 38.901（Release 16）：Study on channel model for frequencies from 0.5 to 100 GHz**  
年份：2020（R16 版本，后续版本可在文中说明“持续演进”）  
来源：ETSI/3GPP 技术报告（行业基准模型）  
核心贡献：提供统一的统计信道建模框架，覆盖多种场景下的**簇化多径、角度/时延扩展、LOS/NLOS 条件与参数生成流程**，是讨论“城市多径导致定位退化”的标准化依据。citeturn15view1  
可用于第一章哪一段：1.2.2 “城市传播机理与误差来源”段（定义/术语与标准引用）。  
它能支撑的观点：城市环境的测距/测角误差不是随机噪声而已，而是与**簇结构、时延扩展、角扩展、遮挡状态**系统相关。

**[8] Modeling air-to-ground path loss for low altitude platforms in urban environments**  
年份：2014（经典奠基）  
来源：IEEE GLOBECOM  
核心贡献：提出低空平台空地链路的统计路径损耗模型，强调**仰角相关**与城市统计参数对 LOS/NLOS 与损耗分布的影响，是 UAV 观测几何影响传播质量的常用基线。citeturn31search10  
可用于第一章哪一段：1.2.2 中“UAV 空地链路：仰角—遮挡—损耗”的基线段。  
它能支撑的观点：UAV 高度与观测仰角不仅改变几何精度（GDOP），也通过 LOS 概率改变**观测可用性与误差分布**。

**[9] On Modeling Satellite-to-Ground Path-Loss in Urban Environments**  
年份：2021  
来源：IEEE Communications Letters  
核心贡献：面向卫星-地面链路给出城市环境路径损耗/遮挡建模框架，强化了“城市几何导致 LOS 概率与损耗统计改变”的论证，对 LEO 终端相关链路很贴合。citeturn33search1  
可用于第一章哪一段：1.2.2 中“卫星/非地面网络在城市环境的衰落与遮挡”段。  
它能支撑的观点：LEO 相关链路在城市环境下同样受“可见天区/遮挡概率”支配，导致信号质量与可探测性高度非均匀。

**[10] TDOA-based localization with NLOS mitigation via robust model transformation and neurodynamic optimization**  
年份：2021  
来源：Signal Processing（Elsevier）  
核心贡献：针对 TDOA 在 NLOS 条件下的鲁棒定位，提出以 ℓ1 等鲁棒准则与优化求解实现对 NLOS 偏差的缓解，体现了“**从 TDOA 指标直接鲁棒化**”的代表路线。citeturn34view0  
可用于第一章哪一段：1.2.2 末尾“经典定位算法在 NLOS 下的退化与典型缓解策略”段。  
它能支撑的观点：NLOS 偏差会造成系统性定位误差，必须引入**鲁棒损失/约束与可实现的求解策略**，而不能仅靠均值滤波。

**[11] Power Allocation and Parameter Estimation for Multipath-based 5G Positioning**  
年份：2021  
来源：IEEE Transactions on Wireless Communications  
核心贡献：把多径从“干扰”提升为“可利用信息”，讨论在存在时钟不同步时如何通过多径辅助定位/解析时钟偏差，并从 CRLB/参数估计角度给出系统设计启示。citeturn35view1  
可用于第一章哪一段：1.2.2 中“多径可利用化：从抑制到利用”的提升段。  
它能支撑的观点：在复杂城市环境里，“抹掉多径”未必最优；合理建模与参数估计可让 NLOS 路径提供额外信息，尤其在存在时钟偏差时。

**[12] Impact of Altitude, Bandwidth, and NLOS Bias on TDOA-Based 3D UAV Localization: Experimental Results and CRLB Analysis**  
年份：2025（ICC Workshops；同时有 arXiv 版本）  
来源：IEEE ICC Workshops / arXiv  
核心贡献：用真实飞行与基准分析展示 TDOA 在混合 LOS/NLOS 下的误差变化规律，指出高度、带宽与 NLOS 偏差对精度的实证影响，是“城市/遮挡导致经典 TDOA 退化”的强证据。citeturn39search0turn39search8  
可用于第一章哪一段：1.2.2 “UAV 平台 + 城市遮挡对 TDOA 的影响”段（用于强化论证的实验型文献）。  
它能支撑的观点：经典 LOS 假设在城市环境不成立；高度与带宽可部分缓解，但仍需要算法层面的自适应与鲁棒机制。

### 共同不足（面向你的选题聚焦）

传播与定位退化研究多以地面基站/车辆/室内为对象，或假定传感器阵列与时间同步较理想；而在你的场景中，“低成本 UAV + 非合作发射源 + 城市 NLOS + 可能间歇发射”会引入更复杂的**观测缺失、偏差非平稳、统计分布随航迹变化**问题，使得固定权重 EKF/LS 等方法很难长期鲁棒。citeturn15view1turn34view0turn39search0

### 第一章如何承接到你的研究

建议以“问题收束”方式承接：城市环境导致 RSSI/AoA/TOA/TDOA 等观测呈现**强非高斯、强相关、强非平稳**的误差结构，因此你不会仅讨论单一观测模型，而是把城市传播退化视为“融合权重与观测可信度应随时空自适应变化”的根本动机，为第三章的自适应融合与强化学习动态加权埋伏笔。fileciteturn0file0

---

## 无人机机动感知、空间采样、Radio Map/REM 重构

### 发展脉络总结（100–200字）

UAV 参与无线感知的研究从“空中采点”逐步发展为“机动传感器阵列 + 主动采样 + 环境电磁数字孪生”：早期多用插值/高斯过程重构二维 RSS 场，随后转向结合城市语义/遮挡机理的环境感知 Radio Map，并引入主动学习以显著减少采样成本；近两年生成式模型（如扩散模型）开始用于稀疏测量下的 REM 超分辨重构，推动多目标定位与识别的可扩展建模。适合挂引文：[13]–[18]。citeturn42search0turn42search19turn36view1turn20view0turn36view3turn16search4

### 推荐文献列表（4–8篇）

**[13] Toward(s) Environment-Aware 6G Communications via Channel Knowledge Map**  
年份：2021（IEEE Wireless Communications；arXiv 2020 可读版）  
来源：IEEE Wireless Communications  
核心贡献：提出 CKM（Channel Knowledge Map）概念，把“无线环境知识”组织成可查询、可更新的地图/模型，为后续 REM/Radio Map 重构与环境感知定位提供统一叙事框架。citeturn42search0  
可用于第一章哪一段：1.2.3 开头“REM/Radio Map 与环境感知通信/定位的概念演进”段。  
它能支撑的观点：Radio Map/REM 不只是工程插值工具，而是迈向“环境感知通信与感知融合”的基础设施。

**[14] A Tutorial on Environment-Aware Communications via Channel Knowledge Map for 6G**  
年份：2024（COMST；arXiv 2023）  
来源：IEEE Communications Surveys & Tutorials（以正式出版信息为准）  
核心贡献：系统总结 CKM 的构建范式（模型驱动/数据驱动/混合）、应用方式与开放问题，是写“研究现状综述”的高质量二级文献。citeturn22search2turn42search19  
可用于第一章哪一段：1.2.3 “国内外研究现状综述”中作为综合综述引用（并可用于组织段落结构）。  
它能支撑的观点：CKM/REM 的关键矛盾在于**高维、稀疏、非平稳**，需要把采样、重构与应用任务联动设计。

**[15] UAV-aided Radio Map Construction Exploiting Environment Semantics**  
年份：2021（arXiv 可读版；对应 IEEE TWC 期刊版为 2023，见 DOI）  
来源：arXiv / IEEE Transactions on Wireless Communications  
核心贡献：提出把城市遮挡机理“语义化”为虚拟障碍/多类别遮挡，从 RSS 测量中联合估计环境语义与 Radio Map，相比纯插值更省样本、更可解释，且天然适配 UAV 空地链路。citeturn36view1turn18search9  
可用于第一章哪一段：1.2.3 “面向城市遮挡的语义 Radio Map：从插值到环境嵌入”段（代表性方法）。  
它能支撑的观点：在城市环境中，Radio Map 的泛化能力取决于是否显式刻画**遮挡/反射等机理**；仅靠数据驱动插值往往样本效率低且迁移困难。

**[16] Bayesian Active Learning for Sample Efficient 5G Radio Map Reconstruction**  
年份：2024  
来源：IEEE Transactions on Wireless Communications  
核心贡献：把 Radio Map 重构与“采样成本/路径成本”显式耦合，以 GP/不确定性为核心进行主动学习选点，体现 UAV/测量车场景下“少量高价值采样”的主流趋势。citeturn20view0turn18search17  
可用于第一章哪一段：1.2.3 “空间采样：从均匀采点到不确定性驱动的主动采样”段。  
它能支撑的观点：采样策略与重构精度强耦合；主动学习可显著降低测量量级，为多目标场景的 REM 实用化铺路。

**[17] UAV-aided Joint Radio Map and 3D Environment Reconstruction using Deep Learning Approaches**  
年份：2022  
来源：IEEE ICC 2022  
核心贡献：把 Radio Map 与三维环境重构联动，代表“无线—环境联合建模”方向，契合你后续要做的多目标信号场重构与识别。citeturn36view3  
可用于第一章哪一段：1.2.3 “Radio Map 与三维环境/数字孪生耦合建模趋势”段。  
它能支撑的观点：面向复杂城市，单纯重构 RSS 场不足以支撑可解释定位；需要走向“电磁场 + 环境结构”的联合学习。

**[18] Physics-Informed Diffusion Model for Radio Environment Map Reconstruction from Sparse Measurements**  
年份：2025  
来源：APCC 2025  
核心贡献：用物理先验约束扩散生成模型，在稀疏测量条件下实现高分辨率 REM 重构，代表生成式 AI 在 REM 超分辨中的新趋势。citeturn16search4  
可用于第一章哪一段：1.2.3 末尾“最新进展与不足：生成式模型 + 物理先验”段（近两年亮点）。  
它能支撑的观点：在“测量稀疏 + 场强分布非平稳”的约束下，生成式模型具备更强的分布学习与超分辨潜力，但对可靠不确定性与跨城迁移仍不充分。

### 共同不足（面向你的选题聚焦）

多数 REM/Radio Map 工作服务于网络覆盖预测或一般定位，较少面向“**非法终端排查**”所需的：多目标同时存在导致的信号叠加可分离性、对抗式间歇发射导致的时变场、以及低成本 UAV 载荷下对“测量噪声、姿态误差、掉点缺测”的系统鲁棒设计。citeturn36view1turn20view0turn16search4

### 第一章如何承接到你的研究

建议在该节末尾明确你的“方法学选择”：你将把 Radio Map/REM 视为多目标定位的**中间表示层**——先用 UAV 机动采样重构空间信号场，再在信号场上做候选终端的定位识别；此外，你的场重构不只追求 RMSE，还要服务于“目标可分辨与可定位”的任务指标，从而自然引出第四章的多目标定位与鲁棒协同。fileciteturn0file0

---

## 强化学习与知识蒸馏在智能定位与鲁棒感知中的应用

### 发展脉络总结（100–200字）

强化学习在定位领域的落地路径通常有两类：其一将“权重/策略选择”建模为 MDP，用 DQN 等方法在离散决策中自适应调度观测或调参；其二面向连续控制，用 DDPG/TD3/SAC 等策略梯度方法学习连续动作（如机动策略或连续权重）。与此同时，知识蒸馏从模型压缩扩展到跨场景迁移与缺失信息补偿，在城市无线定位中被用于把“高成本教师能力”迁移到“轻量学生模型”，以适配边缘设备与掉点鲁棒推理。适合挂引文：[19]–[26]。citeturn37search0turn38search0turn38search1turn37search2turn28view2turn28view0turn28view1turn37search1

### 推荐文献列表（4–8篇）

**[19] Human-level control through deep reinforcement learning（DQN）**  
年份：2015（经典奠基）  
来源：Nature  
核心贡献：提出 DQN，把深度网络与 Q-learning 结合，奠定“离散动作策略学习”的范式，是后续 DQN 类动态加权/策略选择的基础引用。citeturn37search0  
可用于第一章哪一段：1.2.4 开头“强化学习方法谱系与引入动机”段（只需一句话点到即可）。  
它能支撑的观点：复杂环境下的自适应决策可通过“价值函数近似 + 经验回放”等机制学习得到，为动态加权提供算法土壤。

**[20] Continuous control with deep reinforcement learning（DDPG）**  
年份：2015  
来源：NeurIPS 2015 / arXiv  
核心贡献：给出面向连续动作空间的 actor–critic 框架（DDPG），为“连续权重/连续机动策略”的学习奠定经典方法线。citeturn38search0  
可用于第一章哪一段：1.2.4 中“从离散到连续：权重/机动策略的连续化学习”段。  
它能支撑的观点：当融合权重被建模为连续变量时，需采用策略梯度/actor–critic 路线而非仅 DQN。

**[21] Addressing Function Approximation Error in Actor-Critic Methods（TD3）**  
年份：2018  
来源：ICML 2018 / arXiv  
核心贡献：针对 actor–critic 的过估计与不稳定问题提出 TD3（双 Q、延迟更新、目标平滑），是“连续控制更稳健”的代表算法。citeturn38search1  
可用于第一章哪一段：1.2.4 中“鲁棒性与稳定性：改进型 actor–critic”的方法论段。  
它能支撑的观点：连续策略学习易受函数逼近误差影响，必须采用更稳健的训练机制以提升泛化与稳定性。

**[22] Soft Actor-Critic（SAC）**  
年份：2018  
来源：arXiv / PMLR  
核心贡献：最大熵 RL 框架下的离策略 actor–critic（SAC），以更稳定的训练性质与样本效率常被用于复杂连续控制问题。citeturn37search2turn37search6  
可用于第一章哪一段：1.2.4 中“最大熵与鲁棒策略”的简述段（为你选择算法提供合理性）。  
它能支撑的观点：最大熵目标可提升探索与策略鲁棒性，适合不确定环境下的连续决策（如动态权重调整）。

**[23] Geometry Adaptive Deep Q-Network for UAV-Based Emitter Localization in Cluttered RF Environments**  
年份：2026  
来源：IEEE（会议论文集；以论文页眉为准）  
核心贡献：面向多径/杂波 RF 环境，用 DQN 驱动 UAV（或多 UAV）基于 TDOA 等不确定性指标自适应改变几何构型，实现更快收敛与更低误差，直接贴合“UAV 机动感知 + 发射源定位”。citeturn28view2  
可用于第一章哪一段：1.2.4 “RL 用于机动感知与发射源定位”的应用段（高度相关）。  
它能支撑的观点：RL 不只是“权重调节”，也可用于“机动采样策略/阵列几何优化”，并在多径环境中优于固定启发式。

**[24] Adaptive target localization under uncertainty using Multi-Agent Deep Reinforcement Learning with knowledge transfer**  
年份：2024（期刊标注为 2024；页面显示 2025 亦可在文中统一按正式出版年）  
来源：The Internet of Things（Elsevier）  
核心贡献：将目标定位问题建模为多智能体 DRL（PPO 等）并显式考虑不确定性（虚警、不可达等），再用迁移学习/知识转移提升定位推断，代表“多智能体协同定位”的一条成熟写法。citeturn28view0  
可用于第一章哪一段：1.2.4 “多智能体协同与不确定环境鲁棒定位”段。  
它能支撑的观点：在不完全可观测与不确定环境下，多智能体策略学习能把“搜索—判定—定位”做成一体化闭环。

**[25] Transfer Learning with Knowledge Distillation for Urban Localization Using LTE Signals**  
年份：2024  
来源：IEEE VTC2024-Fall  
核心贡献：在城市 LTE 信号定位中引入迁移学习 + 知识蒸馏，体现“把教师模型/源域知识迁移到目标域/轻量学生”以增强城市定位泛化能力的代表实践。citeturn28view1  
可用于第一章哪一段：1.2.4 “知识蒸馏用于城市无线定位迁移/轻量化”的主论段。  
它能支撑的观点：城市无线定位面临跨区域分布漂移，蒸馏可作为边缘部署与跨域泛化的有效工具链。

**[26] Distilling the Knowledge in a Neural Network（Knowledge Distillation）**  
年份：2015（经典奠基）  
来源：arXiv  
核心贡献：提出经典 teacher–student 蒸馏框架，是你后续“多无人机/掉点条件下知识蒸馏鲁棒协同定位”的根文献。citeturn37search1  
可用于第一章哪一段：1.2.4 中“知识蒸馏基本范式与定位任务映射”的定义段（用一句话即可）。  
它能支撑的观点：教师—学生的知识迁移能把“高算力/多信息模型”的能力转移到“低算力/缺测模型”，适合掉点鲁棒。

### 共同不足（面向你的选题聚焦）

定位领域的 RL/KD 应用普遍存在三类共性不足：  
一是把复杂误差来源粗化为状态特征，缺少与传播机理/观测物理一致的状态—奖励设计；二是训练环境与真实城市电磁环境分布差异大，策略泛化缺乏可验证边界；三是蒸馏多强调压缩与迁移，较少与“多机协同 + 掉点缺测 + 多目标 REM”形成端到端系统闭环。citeturn28view2turn28view0turn28view1turn37search1

### 第一章如何承接到你的研究

建议承接逻辑写成“你的必要性”：  
由于城市 NLOS 的观测可信度随位置/高度/姿态快速变化，固定权重融合很难鲁棒，因此你将在第三章引入 RL（以 DQN 类为主）学习动态权重与观测调度；又由于多无人机与掉点缺测不可避免，你将在第四章把 teacher–student 蒸馏用于协同定位的轻量化与缺测鲁棒推理，从而构建“学习型鲁棒融合 + 学习型协同”的完整链路。fileciteturn0file0

---

## 第一章最适合引用的核心文献清单与段落化引用建议

### 最适合第一章引用的 15–25 篇核心文献清单（建议按下列顺序编号）

1) [1] LEO-SoOP/PNT 总体综述：LEO 机会信号用于定位授时的研究版图与挑战。citeturn4view1  
2) [2] Starlink 下行结构解析里程碑：为非合作感知和可相关处理提供结构基础。citeturn1view0  
3) [3] Starlink 载波相位定位首批结果：证明高精度相位观测可提取。citeturn10view1  
4) [4] Starlink 类 OFDM 结构揭示：结构化参考信号推动精确定位。citeturn6view0  
5) [5] Starlink for PNT 工程化阐述：信号、处理链路与能力边界讨论。citeturn6view1  
6) [6] Starlink 定时特性与定位影响：指出时钟不连续/抖动等关键限制。citeturn6view2  
7) [7] 3GPP TR 38.901：城市多径/角扩展/时延扩展等误差源的标准化建模依据。citeturn15view1  
8) [8] UAV 空地链路经典路径损耗模型：仰角相关 LOS 概率与损耗基线。citeturn31search10  
9) [9] 卫星到地面城市路径损耗建模：强调城市遮挡与可见天区对链路统计的支配作用。citeturn33search1  
10) [10] TDOA 在 NLOS 下的鲁棒定位：代表性鲁棒化思路与可实现求解。citeturn34view0  
11) [11] 多径可利用化定位：在时钟不同步等条件下利用多径提升定位。citeturn35view1  
12) [12] UAV 场景 TDOA 退化的实证研究：高度/带宽/NLOS 偏差对误差影响。citeturn39search0turn39search8  
13) [13] CKM 概念奠基：把环境知识组织成地图/模型，支撑 REM/Radio Map 叙事。citeturn42search0  
14) [14] CKM 综述教程：总结构建与应用框架及开放问题。citeturn22search2turn42search19  
15) [15] 语义 Radio Map（UAV 友好）：从 RSS 重构遮挡语义并提升样本效率。citeturn36view1turn18search9  
16) [16] 主动学习采样：用不确定性驱动的选点减少测量成本。citeturn20view0turn18search17  
17) [17] Radio Map 与 3D 环境联合重构：体现无线—环境联合建模趋势。citeturn36view3  
18) [18] 物理先验扩散模型重构 REM：稀疏测量下高分辨率重构的最新路线。citeturn16search4  
19) [19] DQN 奠基：RL 动态决策（离散动作）方法根文献。citeturn37search0  
20) [22] SAC：最大熵 RL 提升连续决策鲁棒性与稳定性代表。citeturn37search2turn37search6  
21) [23] UAV 发射源定位中的 DQN：在多径环境下学习几何/机动策略。citeturn28view2  
22) [24] 多智能体 RL 定位与知识转移：不确定环境下协同定位范式。citeturn28view0  
23) [25] LTE 城市定位的迁移学习 + 蒸馏：城市无线定位泛化与轻量化实践。citeturn28view1  
24) [26] 知识蒸馏奠基：teacher–student 的基本范式与引用源头。citeturn37search1  

> 注：其中 [19]/[26] 属于“方法奠基文献”，在第一章可用 1–2 句点到即可，避免写成教材式展开。

### 按综述段落组织的引用建议（可直接照此写第一章 1.2）

**第一段：总体研究背景与“非法 LEO 终端定位识别”的问题定位（概述趋势 + 点出挑战）**  
建议引用：[1], [6], [7], [12]  
写法要点：用 [1] 定义研究版图，用 [6]/[7]/[12] 点出“城市遮挡 + 时统不可控 + 经典方法退化”的刚性挑战，为你论文动机定调。citeturn4view1turn6view2turn15view1turn39search0

**第二段：LEO-SoOP 与 Starlink 下行解析的进展脉络（从可行性到结构化处理）**  
建议引用：[2], [3], [4], [5]（必要时补 [1]）  
对比关系建议：  
- “结构解析奠基”对比“相位可行性验证”：用 [2] vs [3] 作对比；  
- “类 OFDM 结构化参考信号”对比“盲跟踪”：用 [4]/[5] 与 [3] 对比。citeturn1view0turn10view1turn6view0turn6view1

**第三段：现有 Starlink/LEO 机会信号定位的关键不足（为什么不能直接搬到非法终端定位）**  
建议引用：[6], [1]  
写法要点：突出“时间基准不受控、观测不连续、工程条件多变”，引出你需要鲁棒融合与机动采样。citeturn6view2turn4view1

**第四段：城市传播机理如何导致 RSSI/AoA/TOA/TDOA 等观测退化（误差来源与统计特征）**  
建议引用：[7], [8], [9]  
写法要点：用 [7] 给标准化误差源术语，用 [8] 解释 UAV 仰角与遮挡对空地观测的影响，用 [9] 把“卫星相关链路”也纳入城市遮挡叙述，增强与你选题的贴合度。citeturn15view1turn31search10turn33search1

**第五段：经典定位方法在 NLOS 下的代表性应对（鲁棒估计与“多径可利用化”两条线）**  
建议引用：[10], [11], [12]  
对比关系建议：  
- “鲁棒化抑制 NLOS 偏差” vs “把多径当信息”：用 [10] vs [11] 对比；  
- “理论/仿真” vs “真实飞行验证”：用 [11]（理论/设计）对比 [12]（实证）。citeturn34view0turn35view1turn39search0

**第六段：UAV 机动感知 + Radio Map/REM 的研究现状（采样—重构—应用闭环）**  
建议引用：[13], [14], [15], [16], [17]（如写生成式进展再补 [18]）  
对比关系建议：  
- “概念与框架” vs “具体可落地方法”：用 [13]/[14] 对比 [15]/[16]/[17]；  
- “环境语义嵌入” vs “不确定性驱动采样”：用 [15] vs [16]。citeturn42search0turn22search2turn36view1turn20view0turn36view3

**第七段：智能定位方法（RL + KD）在鲁棒感知中的应用与不足（引到你的第三/四章）**  
建议引用：[19], [22], [23], [24], [25], [26]  
对比关系建议：  
- “单智能体/离散动作（DQN）” vs “更鲁棒连续控制（SAC）”：用 [19] vs [22]；  
- “RL 用于机动/几何优化” vs “RL 用于融合权重”：可用 [23] 作为机动代表，再在你论文里承接到权重学习（不必强行找同类文献堆砌）；  
- “蒸馏用于城市无线定位迁移” vs “蒸馏用于掉点鲁棒协同”：用 [25]（迁移）对接你自己的掉点场景，并用 [26] 给出范式根引用。citeturn37search0turn37search2turn28view2turn28view0turn28view1turn37search1

**第八段：总结不足并引出你的研究贡献点（第一章收束段）**  
建议引用：择要引用 [1], [6], [7], [15], [16], [25] 再加你自己的“本文工作概述”。citeturn4view1turn6view2turn15view1turn36view1turn20view0turn28view1  
写法要点：把不足收束为三条“必须解决的矛盾”，并逐条对应你的章节贡献（第三章自适应融合与 RL 动态权重；第四章多目标 REM 重构 + KD 鲁棒协同）。fileciteturn0file0
