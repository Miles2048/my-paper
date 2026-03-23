# 基于低成本无人机平台的非法低轨卫星终端定位识别：第一章研究现状与第二章理论基础深度文献调研与引用建议

## 选题边界与筛选标准

本次调研以你论文《基于低成本无人机平台的非法低轨卫星终端定位识别算法研究》的技术主线为中心，聚焦“空中平台（低成本无人机）+ 城市复杂传播 + 非合作/无源定位 + 多源观测融合（RSSI/AoA/TOA/TDOA/PDR）+ 强化学习动态权重 + Radio Map/REM 重构 + 多无人机/掉点鲁棒协同（知识蒸馏）”这一闭环链路，并将输出组织为可直接写入硕士论文第一章（研究现状）与第二章（系统模型与理论基础）的内容与参考文献框架。fileciteturn0file0

按你的额外要求，本文献清单已主动排除下列低相关方向：普通 WiFi 室内定位（且不涉及城市复杂环境或无人机平台）、纯视觉目标检测（且不涉及无线感知/Radio Map/定位）、与定位观测或空间采样无关的一般路径规划、以及不涉及非合作信号感知/定位/终端排查的一般卫星通信系统优化研究。

## 第二章系统模型与定位理论基础：按小节分类的文献与“公式支撑点”映射

这一部分的目标是：你在第二章写到任何一个系统模型假设、观测方程、误差源、性能下界（CRLB）或算法基线时，都能明确“该引用支撑哪个公式/概念”，并知道“哪些必引、哪些可少引”。

### 非合作式无源定位系统模型

**写作抓手**：你第二章通常需要给出“状态—观测—噪声”的统一形式，例如  
- 状态（目标）位置：\(\mathbf{p}\in\mathbb{R}^2/\mathbb{R}^3\)，或含速度、钟差等扩展状态；  
- 观测：TDOA/TOA/AoA/RSSI（以及必要时的 Doppler/FDOA）；  
- 噪声：高斯/混合噪声/含偏置（NLOS bias）模型；  
- 估计：WLS/MLE/EKF/粒子滤波；  
- 性能界：FIM/CRLB 与可观测性讨论。

**经典基础文献（建议“必引”，支撑系统模型主干与可观测量定义）**  
- [1] 可直接支撑“无源定位系统的统计建模框架、TDOA/FDOA 等经典无源体制与误差传播”的第二章系统模型总述，以及“以几何曲线（双曲线/双曲面）描述 TDOA 约束”的推导脉络。citeturn20search5  
- [2] 可直接支撑你在 TDOA 场景下把非线性双曲线方程转化为可解的两步/加权最小二乘（WLS）形式，并可用于说明“在小误差区间可逼近 CRLB”的常见论述（也是很多后续工作引用的经典基线）。citeturn4search6  
- [6] 可支撑“Direct Position Determination, DPD（直接定位/直达定位）相对于两步法在弱信号与模型失配时的优势”这一系统模型分支（尤其适合写入“非合作、弱信号、噪声/多径较强”背景下的基线对比）。citeturn20search21turn20search6  
- [49] 若你需要把“远场被动源定位”写得更严谨，可用于支撑 TDOA/FDOA 的几何结构、可观测性与渐近分析（适合放在“建模假设与几何解释”小节）。citeturn2search17  

**近年补充文献（建议“择要引用”，支撑你论文贴近“无人机机动平台/复杂环境”的论述）**  
- [13] 适合支撑“在城市环境中，多径不仅是误差源，也可被显式利用（multipath exploitation）形成额外 TOA 约束”的研究分支，用于引出你后续的 Radio Map/RT（射线追踪）思路。citeturn5search22  
- [14] 适合支撑“将射线追踪产生的多径指纹/信道冲激响应（CIR）用于城市 NLOS 场景下的定位增强”的方法链路（可作为你第二章中“环境先验/地图辅助”的早期代表）。citeturn15search3  

**本小节最需要引用的点**：TDOA/FDOA/AoA 等无源观测的系统化统计建模（必引 [1]）；TDOA WLS 基线与近 CRLB 经典结论（必引 [2]）；在弱信号/模型失配时直接定位 DPD 的合理性（建议引 [6]）。citeturn20search5turn4search6turn20search21  
**可少引的点**：若你的第二章不准备展开 DPD 的工程实现与复杂度，可只在“相关方法”中短引 [6]，并把主笔墨放在你自己的融合与鲁棒策略上。

### 复杂城市环境下多径、NLOS、时延扩展与衰落

**写作抓手**：第二章里你通常要把“城市复杂传播导致定位退化”的机理写清楚，包括  
- 多径与 NLOS 导致的 TOA/TDOA 正偏（bias）与非高斯误差；  
- RSSI 的阴影衰落（log-normal shadowing）与时间/空间相关性；  
- 时延扩展（RMS delay spread）导致的宽带测距分辨率与同步误差；  
- 城市街谷/遮挡导致 LOS 概率下降与观测可用性降低。

**经典/标准文献（建议“必引”，用于给出“权威信道统计模型”）**  
- [10] 可作为你第二章城市信道统计与参数定义的权威来源：覆盖 UMi/UMa 等场景的路径损耗、阴影衰落、延迟扩展等模型族，非常适合用于你“复杂城市环境信道模型假设”小节的主引用。citeturn2search27  
- [9] 可用于系统性支撑“TOA/TDOA 在 NLOS 下的偏置问题与典型缓解路径（检测、剔除、鲁棒估计、统计建模）”，适合写在“经典定位退化机理与对策综述”中。citeturn5search0  

**代表性补充文献（建议“重点引用 1–2 篇”，用于把‘城市 NLOS’写得更贴近你题目）**  
- [14]（同上）把“射线追踪 + 多径指纹 + 学习器”作为城市 NLOS 场景增强手段，可用于支撑你后续 Radio Map/REM 重构章节的技术动机。citeturn15search3  
- [13] 强调“单站/少站也可利用多径结构定位”，用于说明“城市多径并非只能回避，还可以被建模为可用信息”。citeturn5search22  

**本小节最需要引用的点**：用 [10] 给出权威统计信道模型（必引）；用 [9] 建立 NLOS 偏置对定位的“共识性结论与分类”。citeturn2search27turn5search0  
**可少引的点**：如果你第二章不打算展开“具体NLOS检测器/分类器”，则无需在第二章堆叠大量 ML-NLOS 论文；把重点放在你第三/四章中“用 RL/REM/KD 做鲁棒化”的创新点即可。

### Starlink 与 LEO Ku 波段下行链路信号结构

你提出“Starlink / LEO Ku 波段下行链路信号结构”作为第二章基础内容，核心是为第三章/第四章做铺垫：你需要说明“可重复/可预测结构（导频/参考信号/帧结构）存在，因此可以从‘非合作’信号中提取可观测量（如多普勒、载波/码相位、相关峰、甚至DOA）”。

**频段与体制边界（建议必引监管/标准材料，避免口径争议）**  
- [15] 与 [16] 适合支撑你对 Starlink Ku 频段“用户下行/上行”的论文式描述（例如用户终端下行 10.7–12.7 GHz、上行 14.0–14.5 GHz 等，同时强调这是监管文件/系统申请材料中的链路频段表述）。citeturn21search7turn21search3turn21search2  

**Starlink 下行信号结构与可观测量（建议作为“第二章必引主干”）**  
- [18] 可直接支撑“Starlink 下行存在 OFDM-like 参考信号/结构，能估计帧长度并进行盲获取与跟踪”的关键论断；非常适合在第二章写出“为什么能从非合作 Starlink 信号中提取同步与跟踪所需结构”。citeturn19search0  
- [19] 适合支撑“从理论到实验的完整链路：如何利用 Starlink 信号实现 PNT（包含信号结构、可观测量生成、误差建模、实验验证）”；你可将其作为“信号结构+可观测量+误差源”的中心引用。citeturn18search0  
- [20][21] 适合支撑你写入“可预测元素/导频、时钟与时频稳定性等‘工程细节’会显著影响可观测量质量（如多普勒/载波相位）”，从而自然引出你后续的鲁棒融合与动态权重（RL）需求。citeturn18search14turn0search20  

**本小节最需要引用的点**：监管材料给出频段边界（必引 [15]/[16]）；信号结构可解析性（必引 [18] 或 [19]）；时频/导频可预测性（建议引 [20][21]）。citeturn21search7turn21search2turn19search0turn18search0turn18search14turn0search20  
**可少引的点**：若你第二章不展开接收机实现细节（同步环、解调链），可把 [20][21] 作为“工程影响因素”点到为止，重点保留“结构存在→可观测量可提取→定位可行”的论证主链。

### RSSI / AoA / TOA / TDOA / PDR 的经典定位模型

你第二章要做的是“把多源观测写成统一观测模型”，并配套给出经典算法与误差来源。下列文献组合能让你写得既“教科书式严谨”，又不至于跑题到普通室内 WiFi 指纹。

**RSSI（路径损耗/阴影衰落）模型与未知发射功率问题**  
- [26] 可用于支撑 RSS/TOA/AoA 等测量统计模型的统一表述（其定位模型总结与 CRB 讨论很适合作为第二章“基本观测模型”引用）。citeturn10search8  
- [48] 若你需要写 RSSI 量化（低成本硬件常见）对性能下界的影响，可用其支撑“量化 RSSI 定位的 CRLB/信息量分析”。citeturn4search7  

**AoA（阵列测角）模型与 MUSIC/子空间法**  
- [27] 可直接支撑你对 MUSIC 这类子空间 DOA/AoA 估计算法的“经典定义与可分辨性直觉”。citeturn9search6  
- [28] 用于支撑“阵列信号处理与 DOA 估计研究脉络（MUSIC/ESPRIT/MVDR 等）与性能讨论”，适合写在 AoA 基础小节的综述段。citeturn10search3  

**TOA/TDOA（测距/双曲线定位）模型**  
- [2] 作为 TDOA 双曲线定位两步 WLS 的经典基线引用，几乎是写第二章“时间差定位”的最短路径。citeturn4search6  
- [9] 用于支撑“TOA/TDOA 在 NLOS 下的偏置与缓解”，与 [2] 配合写“模型—退化—对策”。citeturn5search0  

**PDR（Pedestrian Dead Reckoning）与漂移机理**  
- [29] 是写 PDR 综述段落的高被引权威来源，适合支撑“PDR 误差随时间累积、需外部绝对观测校正”的共识表述。citeturn9search4  

**本小节最需要引用的点**：TDOA 经典基线公式与两步 WLS（必引 [2]）；AoA/MUSIC 的经典来源（必引 [27] 或 [28]）；PDR 漂移与校正需求（必引 [29]）；RSSI 统计测量模型可用 [26] 统一引用。citeturn4search6turn9search6turn9search4turn10search8  
**可少引的点**：若你不准备在第二章深入阵列校准/互耦/宽带 DOA 的工程细节，则 [28] 可仅在综述段落引用一次即可，把推导重点放在你论文用到的 AoA 观测方程与误差模型。

### 马尔可夫决策过程、DQN 强化学习基础与知识蒸馏基础

你第二章写 RL/DQN 与 KD 的关键，是“只写与你后续算法直接相关的概念”：MDP 五元组、价值函数/贝尔曼方程、DQN 的经验回放与目标网络；KD 的软标签/温度、KL 散度损失、teacher–student 训练范式，以及在多智能体/掉点场景下的知识迁移动机。

**MDP/RL/DQN（建议必引）**  
- [32] MDP、价值函数、策略、贝尔曼方程等基础定义的权威教材来源，适合支撑你第二章“MDP 建模”与“Q 学习/价值迭代”基础段落。citeturn3search36  
- [34] DQN 的里程碑工作，适合支撑你第二章“为什么要用深度网络逼近 Q 函数、经验回放与目标网络的稳定化动机”。citeturn3search26  
- [33] 若你希望把 MDP 的理论写得更严谨（如收敛性、动态规划框架），可引用该书作为补强。citeturn3search24  

**知识蒸馏 KD（建议必引 1–2 篇）**  
- [35] 经典 KD 起点文献（teacher–student/soft targets/temperature），适合支撑你第二章 KD 基础段落的核心公式（KL/交叉熵组合）。citeturn3search35  
- [36] 若你希望把 KD 的“范式扩展（logits/feature/关系蒸馏、自蒸馏等）”写得更系统，可引用该综述作为补充。citeturn7search19  

**面向多智能体/协同鲁棒的 KD 代表（建议用于第四章/多无人机掉点动机铺垫）**  
- [37][38] 可用于支撑“KD 不仅用于压缩，也可用于多智能体知识迁移/训练加速/策略复用”的论述，为你“多无人机或掉点条件下的鲁棒协同定位（知识蒸馏）”提供学术锚点。citeturn7search16turn7search0  

**本小节最需要引用的点**：MDP 基本定义与贝尔曼方程（必引 [32]）；DQN 稳定训练机制（必引 [34]）；KD 的软标签与温度/损失形式（必引 [35]）。citeturn3search36turn3search26turn3search35  
**可少引的点**：如果你的论文创新不在 KD 机制本身，而在“用 KD 抵抗无人机掉点/协同鲁棒”，则 [36] 可少引，把重点放在你自己的蒸馏目标设计与鲁棒性实验。

### 定位误差指标与 CRLB

你第二章一般需要两类内容：  
- 误差指标：RMSE/MAE/CEP、误差椭圆、GDOP/几何精度；  
- 理论下界：Fisher 信息矩阵（FIM）与（B）CRLB，用于阐述“观测配置、噪声方差、几何关系如何决定极限性能”。

**经典必引与可直接支撑公式推导的文献**  
- [3] 适合支撑你第二章 CRLB/FIM 的标准定义、向量参数 CRLB、参数变换下的 CRLB 推导模板。citeturn4search0  
- [4] 适合支撑更一般的估计理论与（Bayesian）van Trees 类界的引用（尤其当你要讨论“先验/随机参数/跟踪”的下界）。citeturn4search25  
- [2] 支撑“经典 TDOA 两步 WLS 在小误差区间可达 CRLB”的常见结论表达（作为算法基线与理论极限之间的桥）。citeturn4search6  
- [47] 若你要讨论“网络定位/锚点不足/尺度效应”的 CRB 结构，可用其作为定位下界分析的经典补充。citeturn4search35  

**本小节最需要引用的点**：CRLB/FIM 公式与推导模板（必引 [3] 或 [4]）；将特定定位算法与 CRLB 对齐的经典结论（建议引 [2]）。citeturn4search0turn4search25turn4search6  
**可少引的点**：GDOP/CEP 等指标可在“实验评价指标”中给出定义并少量引用（通常 1–2 篇足够），把引用额度留给“下界与观测模型”更关键。

## 第一章国内外研究现状：四条技术主线的脉络总结、代表文献与承接写法

下面每条主线先给出一段**可直接放入论文**、约 150–250 字的综述段（强调发展脉络、代表方法与不足），并在段末标注**最适合挂的引用编号**。随后给出 4–8 篇代表性文献（按“能支撑的观点”组织），并总结共同不足与如何承接到你的研究。

### 低轨卫星机会信号利用与 Starlink/LEO 下行链路解析

近年 LEO 巨型星座的 Ku 带宽带信号被系统性验证可作为“卫星机会信号（SoOP）”用于定位定时。研究路线通常从可直接提取的多普勒观测与纯音信号出发，逐步推进到对 Starlink 下行 OFDM-like 参考信号/帧结构与可预测导频元素的盲解析，实现载波/码相位跟踪与多星融合定位；同时，星历与时频误差、平台动态与遮挡对可观测量质量的影响被认为是制约精度与鲁棒性的关键瓶颈，从而推动差分/校正与跨系统融合的发展。建议引用：[18][19][20][21][22][23][24]。citeturn19search0turn18search0turn18search14turn0search20turn19search5turn19search3turn19search22  

**推荐文献（代表性 7 篇）**  
- **[18]** 标题：Unveiling Starlink LEO Satellite OFDM-Like Signal Structure Enabling Precise Positioning；年份：2024；来源：IEEE Transactions on Aerospace and Electronic Systems（letter）。核心贡献：给出 Starlink 下行 OFDM-like RS 的盲解析、帧长度估计、盲获取/跟踪与实验定位误差示例。可用于第一章哪一段：SoOP 从“可用”到“可解析”的关键突破段。它能支撑的观点：Starlink 下行包含可用于同步/跟踪/定位的结构性信号成分。citeturn19search0  
- **[19]** 标题：Unveiling Starlink for PNT；年份：2025；来源：NAVIGATION: Journal of the Institute of Navigation。核心贡献：从理论与实验两侧系统阐述利用 Starlink 信号实现 PNT 的完整链路（结构、观测量、误差来源与实验验证）。可用于第一章哪一段：该方向“系统化与实验落地”里程碑段。它能支撑的观点：Starlink SoOP 已具备可复现实验级定位能力，但精度受星历/时频与环境影响显著。citeturn18search0  
- **[20]** 标题：Timing Properties of the Starlink Ku-Band Downlink；年份：2025；来源：arXiv 预印本。核心贡献：聚焦 Starlink 下行的时频/定时性质，讨论时钟/频偏等对观测量稳定性的影响。可用于第一章哪一段：从“信号结构解析”过渡到“工程误差源”段。它能支撑的观点：时频特性会成为 SoOP 定位精度与稳健性的隐性瓶颈。citeturn18search14  
- **[21]** 标题：Pilots and other predictable elements in the Starlink downlink signal；年份：2026；来源：arXiv 预印本。核心贡献：从“可预测元素”角度进一步揭示可用于同步/估计的信号结构。可用于第一章哪一段：SoOP 接收机“从盲到半盲/弱监督”趋势段。它能支撑的观点：可预测导频/结构的存在使得“非合作”信号仍可产生稳定观测。citeturn0search20  
- **[22]** 标题：Survey on Opportunistic PNT With Signals From LEO Communication Satellites；年份：2025；来源：IEEE Communications Surveys & Tutorials。核心贡献：系统综述 LEO SoOP PNT 的设计要点、误差源与融合趋势。可用于第一章哪一段：发展脉络与研究分支梳理段（权威综述锚点）。它能支撑的观点：LEO SoOP 研究已形成相对完整的“信号处理—系统误差—融合应用”谱系。citeturn19search5turn19search13  
- **[23]** 标题：Toward Massive Satellite Signals of Opportunity Positioning；年份：2024；来源：Space: Science & Technology。核心贡献：面向“海量卫星机会信号定位”总结挑战与方法框架（尤其是未知发射时刻/星历误差等）。可用于第一章哪一段：从 Starlink 个例推广到“海量 LEO SoOP”趋势段。它能支撑的观点：巨型星座带来可用性优势，但同步与先验不完备会重塑算法设计。citeturn19search3  
- **[24]** 标题：Opportunistic Navigation with Doppler Measurements from Iridium NEXT and Orbcomm LEO Satellites；年份：2019（以论文公开版本为准）；来源：公开 PDF（多普勒观测+EKF 融合）。核心贡献：展示多星座 Doppler 观测与滤波融合的可行链路，为后续 Starlink 工作提供“多星融合”先行范式。可用于第一章哪一段：SoOP 早期“多普勒—滤波”路线段。它能支撑的观点：在缺少导航体制设计的情况下，多普勒仍能形成可用定位观测。citeturn19search22  

**共同不足（可作为第一章批判点）**  
多数 SoOP 研究强调信号可解析与实验定位可行，但在“城市遮挡/多径导致观测间歇、低成本接收链路量化误差、以及与无人机机动采样结合的系统建模”方面仍不充分；同时，误差建模往往在接收机侧展开，缺少与“空间采样策略/多源融合鲁棒性”联动的统一框架。citeturn19search5turn18search0turn18search14  

**你在第一章如何承接到自己的研究**  
可用“研究缺口三连”承接：其一，SoOP 可解析性已被证实，但在复杂城市环境下观测可用性与误差分布更恶劣；其二，现有工作多以单一观测量（如 Doppler/相位）为主，你的论文转向“RSSI/AoA/PDR 等多源观测融合”更贴近低成本无人机平台；其三，面对掉点与多目标，你将引入 REM 重构与知识蒸馏来强化鲁棒性，从而把 SoOP—城市传播—无人机采样—智能融合贯通成系统方案。citeturn19search0turn2search27turn13search2turn18search0  

### 复杂城市环境传播机理与经典定位技术退化问题

城市街谷与高密建筑导致 LOS 概率下降，多径与 NLOS 使 TOA/TDOA 出现正偏、AoA 出现角度扩展与误配、RSSI 出现强阴影衰落与空间相关，从而使经典几何定位（WLS/MLE）在噪声增大或偏置存在时明显退化。研究发展呈现两条路线：一条是以标准/统计信道模型刻画误差与可用性；另一条是引入地图/射线追踪等环境先验，把多径从“干扰”转化为“特征/指纹”，并与学习器结合做 NLOS 增强。建议引用：[9][10][2][13][14]。citeturn2search27turn5search0turn4search6turn5search22turn15search3  

**推荐文献（代表性 6 篇）**  
- **[10]** 标题：3GPP TR 38.901（5G NR 信道模型：UMi/UMa 等） ；年份：版本随更新（建议引用你实际采用的版本号）；来源：3GPP Technical Report。核心贡献：给出城市典型场景下路径损耗、阴影衰落、延迟扩展等统计模型，为“城市复杂传播→观测退化”提供权威参数化依据。可用于第一章哪一段：国内外研究现状中“城市传播机理共识/标准化模型”段。它能支撑的观点：城市环境下多径/NLOS 属于可建模的系统性误差源。citeturn2search27  
- **[9]** 标题：A Survey on TOA Based Wireless Localization and NLOS Mitigation Techniques；年份：2009；来源：IEEE Communications Surveys & Tutorials。核心贡献：系统梳理 TOA/TDOA 及 NLOS 偏置缓解的典型方法路径，可作为“经典定位退化问题”的总引用。可用于第一章哪一段：经典定位技术在 NLOS 下退化与典型对策段。它能支撑的观点：NLOS 偏置是时间测量定位退化的主因，且有可归类的缓解策略。citeturn5search0  
- **[2]** 标题：A simple and efficient estimator for hyperbolic location；年份：1994；来源：IEEE Transactions on Signal Processing。核心贡献：TDOA 双曲线定位经典两步 WLS（Chan 算法），并给出小误差区域近 CRLB 的结论。可用于第一章哪一段：经典 TDOA 基线与“理论上可接近下界但对大误差/偏置敏感”的对比段。它能支撑的观点：几何型闭式解在理想噪声下高效，但复杂环境下鲁棒性不足。citeturn4search6  
- **[13]** 标题：Single-sensor RF Emitter Localization based on Multipath Exploitation；年份：2013；来源：公开 PDF（城市多径利用）。核心贡献：提出利用多径 TOA 结构实现单站/少站被动定位的思路，体现“多径可用”的反直觉路线。可用于第一章哪一段：从“抑制多径”到“利用多径”的脉络转折段。它能支撑的观点：城市多径可被转化为额外约束信息，而非纯噪声。citeturn5search22  
- **[14]** 标题：Enhancement of Localization Systems in NLOS Urban Scenario with Multipath Ray Tracing Fingerprints and Machine Learning；年份：2018；来源：Sensors。核心贡献：射线追踪指纹 + 学习框架增强 NLOS 城市定位，连接“环境先验/Radio Map”与“定位增强”。可用于第一章哪一段：地图/射线追踪先验与学习方法结合段。它能支撑的观点：RT/指纹可显著改善 NLOS 场景定位效果。citeturn15search3  
- **[10]/[14]（组合引用写法建议）**：可用于写“标准化统计模型（TR 38.901）负责描述一般规律，而 RT/数据驱动负责描述具体城区细节”的对比关系，从而引出你第四章 Radio Map/REM 重构的动机。citeturn2search27turn15search3  

**共同不足**  
现有城市定位退化研究大量集中在“单一观测（如 TOA）+ NLOS 检测/缓解”或“离线构图指纹”，对“低成本无人机平台的机动采样约束、观测多源异质性（RSSI/AoA/PDR）与动态融合权重学习”讨论不足，尤其缺少把“采样策略—融合鲁棒—定位下界”统一建模的论文式表达。citeturn5search0turn2search27turn15search3  

**你在第一章如何承接**  
可承接为：城市复杂传播使经典定位退化已成共识，因此你将从系统层面同时解决“观测质量不稳定（NLOS/掉点）”与“观测来源异质（RSSI/AoA/PDR）”两类问题：先建立统一观测模型与误差度量（第二章），再通过强化学习实现动态权重与鲁棒融合（第三章），并用 REM/Radio Map 解决多目标与空间场重构（第四章）。citeturn4search6turn2search27turn15search3turn16search1  

### 无人机机动感知、空间采样与 Radio Map/REM 重构

无线电环境图（REM/Radio Map）概念最初服务于认知无线电的环境感知与频谱管理，随后在 UAV 辅助通信与定位中演化为“空间信号场/路径损耗场”的数字孪生表达。研究脉络从早期的分段回归/克里金等插值到深度学习的端到端重构（如 CNN/Transformer/生成模型），并进一步发展出“主动采样/信息驱动航迹规划”以在有限航时下最大化地图重构质量。该方向的关键不足在于：数据分布迁移（城区变化）、采样稀疏与实时性约束下的鲁棒泛化与不确定性评估仍较薄弱。建议引用：[44][45][39][40][41][42][43][46]。citeturn2search8turn2search28turn13search2turn15search2turn15search13turn15search4turn15search27turn6search16  

**推荐文献（代表性 8 篇）**  
- **[44]** 标题：REM-enabled situation-aware cognitive algorithms for NR and WRAN systems；年份：2006；来源：IEEE DySPAN。核心贡献：早期把 REM 作为“环境知识库/地图”用于认知系统决策的代表，为后续 REM 概念在通信与感知中的扩展奠基。可用于第一章哪一段：REM 概念起源与应用外延段。它能支撑的观点：REM 是“可被学习/更新/用于决策”的环境表示。citeturn2search8  
- **[45]** 标题：On the Construction of Radio Environment Maps for Cognitive Radio Networks；年份：2013；来源：IEEE Communications Surveys/或相关期刊（以原文为准）。核心贡献：讨论 REM 构建与应用框架，适合支撑你对 REM 的系统定义。可用于第一章哪一段：REM 构建框架与方法分类段。它能支撑的观点：REM 构建通常涉及测量、融合、插值/学习与更新。citeturn2search28  
- **[39]** 标题：Learning radio maps for UAV-aided wireless networks: A segmented regression approach；年份：2017；来源：IEEE ICC（论文 PDF）。核心贡献：把 UAV 测量稀疏条件下的 Radio Map 学习与分段结构结合，是“UAV + Radio Map（学习）”早期高相关代表。可用于第一章哪一段：UAV 机动采样推动 Radio Map 学习段。它能支撑的观点：UAV 测量可用于重构 Radio Map，且传播结构可分段建模。citeturn13search2  
- **[40]** 标题：RadioUNet: Fast Radio Map Estimation with Convolutional Neural Networks；年份：2021；来源：IEEE Transactions on Wireless Communications。核心贡献：用 CNN 学习城市几何到路径损耗场映射，显著加速 Radio Map 生成，并讨论从仿真到现实的迁移。可用于第一章哪一段：深度学习重构 Radio Map 的代表作段。它能支撑的观点：深度网络可替代昂贵 RT，在城市场景快速预测路径损耗图。citeturn15search2  
- **[41]** 标题：Deep Completion Autoencoders for Radio Map Estimation；年份：2022；来源：IEEE Transactions on Wireless Communications。核心贡献：把 Radio Map 视为可从稀疏测量“补全”的结构化对象，提出深度补全自编码器并强调“从经验学习传播结构”。可用于第一章哪一段：从插值到“可学习的补全”转折段。它能支撑的观点：数据驱动可在更少测量下重构地图，并具经验迁移价值。citeturn15search13turn15search8  
- **[42]** 标题：Spectrum Surveying: Active Radio Map Estimation with Autonomous UAVs；年份：2022；来源：IEEE Transactions on Wireless Communications。核心贡献：显式研究“主动采样/在线更新/信息驱动航迹”的 UAV 频谱测绘范式。可用于第一章哪一段：“地图重构+采样策略联动”的关键发展段。它能支撑的观点：航迹与测点选择是 Radio Map 质量的决定性因素之一。citeturn15search4turn15search8  
- **[43]** 标题：Radio Map Estimation: A Data-Driven Approach to Spectrum Cartography；年份：2022；来源：IEEE Signal Processing Magazine（论文 PDF）。核心贡献：从“频谱制图/Radio Map”角度给出数据驱动方法总结，适合写综述的“方法分类+挑战”段。可用于第一章哪一段：Radio Map 方法谱系化总结段。它能支撑的观点：Radio Map 是空间场重构问题，涉及插值、学习、主动采样与不确定性。citeturn15search27turn15search8  
- **[46]** 标题：A Recent Survey on Radio Map Estimation Methods for …；年份：2025；来源：Electronics（survey）。核心贡献：汇总 Radio Map 数据集与方法分类，适合作为最新综述补强引用。可用于第一章哪一段：近年补充综述与数据集/评测段。它能支撑的观点：Radio Map 研究已形成模型驱动/数据驱动/混合三类主线，并面临数据与泛化挑战。citeturn6search16  

**共同不足**  
1) 许多 Radio Map 工作以通信覆盖预测为主，定位识别尤其是“多目标非法终端排查”场景链路较少；2) 稀疏采样下的不确定性量化与在线更新仍难满足无人机有限航时；3) 仿真到现实迁移与跨城区泛化是持续瓶颈。citeturn15search2turn15search13turn15search4turn6search16  

**你在第一章如何承接**  
你可以强调“REM/Radio Map 在你论文中不是为了通信规划，而是为了多目标定位识别”：把信号场重构作为中间层表示，服务于终端候选区域筛选、关联与身份判别，并在掉点/多无人机条件下用蒸馏/协同机制保证地图与定位的连续性与鲁棒性。citeturn13search2turn15search4turn7search16turn7search0  

### 强化学习与知识蒸馏在智能定位与鲁棒感知中的应用

传统多源融合通常依赖固定权重或基于经验的自适应策略，难以应对城市环境下误差分布非平稳、观测掉点与多目标干扰等问题；因此，强化学习逐渐被用于把“权重/测量调度/协同选择”显式建模为 MDP，在交互中学习策略（如 DQN）。与此同时，知识蒸馏从模型压缩扩展为“知识迁移/协同鲁棒”工具，被用于多智能体知识复用与在缺失模态/掉点条件下训练更稳健的学生模型。该方向的共同挑战在于奖励设计与可解释性，以及跨环境泛化与安全约束。建议引用：[32][34][31][30][35][37][38]。citeturn3search36turn3search26turn16search1turn16search0turn3search35turn7search16turn7search0  

**推荐文献（代表性 7 篇）**  
- **[32]** 标题：Reinforcement Learning: An Introduction (2nd ed.)；年份：2018；来源：MIT Press（教材）。核心贡献：MDP、价值函数、贝尔曼方程与核心算法体系化定义。可用于第一章哪一段：RL 基础“只引用不展开”的方法论定义段。它能支撑的观点：定位融合可被建模为序贯决策问题。citeturn3search36  
- **[34]** 标题：Human-level control through deep reinforcement learning；年份：2015；来源：Nature。核心贡献：DQN 框架（经验回放、目标网络）奠定深度值函数方法基础。可用于第一章哪一段：DQN 作为代表性方法的起点引用段。它能支撑的观点：深度网络可逼近高维状态下的最优 Q 函数。citeturn3search26  
- **[31]** 标题：Decentralized Scheduling for Cooperative Localization With Deep Reinforcement Learning；年份：2019；来源：IEEE Transactions on Vehicular Technology（PDF）。核心贡献：把“测量调度”以 CRLB 为指标写成 RL 问题，是 RL 进入定位系统设计的代表性论文之一。可用于第一章哪一段：RL 用于定位系统“调度/选择”的代表段。它能支撑的观点：RL 可直接优化定位性能指标（如 CRLB 门限达成）。citeturn16search1  
- **[30]** 标题：UAV-Based Interference Source Localization: A Multimodal Q-Learning Approach；年份：2019；来源：IEEE Access（PDF）。核心贡献：将 Q-learning 用于 UAV 干扰源搜索/定位并强调对动态环境的自适应。可用于第一章哪一段：UAV +（无线）定位 + RL 的早期落地工作段。它能支撑的观点：RL 能在未知环境中自适应调整策略以提升定位/搜索效率。citeturn16search0  
- **[35]** 标题：Distilling the Knowledge in a Neural Network；年份：2015；来源：arXiv。核心贡献：提出以软标签/温度为核心的 KD 基本范式。可用于第一章哪一段：知识蒸馏基础定义段。它能支撑的观点：teacher–student 可实现知识传递与轻量化部署。citeturn3search35  
- **[37]** 标题：KnowRU: Knowledge Reuse via Knowledge Distillation in Multi-Agent Reinforcement Learning；年份：2021；来源：开放获取论文（PMC）。核心贡献：KD 用于多智能体知识复用/迁移，体现“KD=协同学习机制”而非仅压缩。可用于第一章哪一段：KD 从压缩走向协同鲁棒的代表段。它能支撑的观点：KD 可缩短训练并增强多智能体策略迁移。citeturn7search16  
- **[38]** 标题：Offline Multi-Agent Reinforcement Learning with Knowledge Distillation；年份：2022；来源：NeurIPS（PDF）。核心贡献：KD 融入离线多智能体学习框架，用于知识转移与性能提升。可用于第一章哪一段：近年 KD+MARL 融合趋势段。它能支撑的观点：KD 可在分布式/异构数据下提升学习效率与性能。citeturn7search0turn7search20  

**共同不足**  
1) 许多 RL 定位相关工作偏重“调度/路径/选择”，但对观测误差机理（NLOS bias、多径结构）与算法可解释性结合不足；2) KD 在定位场景下多被当作泛化“工具箱”，缺少针对掉点/缺测的蒸馏目标与理论分析；3) 实验往往局限仿真或单一场景，跨城区泛化证据不足。citeturn16search1turn16search0turn7search16turn7search0  

**你在第一章如何承接**  
你可以把创新承接写成“问题—方法—优势”链：城市复杂环境下观测质量非平稳 → 固定权融合不足 → 以 MDP+DQN 学习动态权重/置信度调度（第三章）；多无人机/掉点导致协同信息不完整 → 以 KD 把“全信息教师”知识压到“掉点可用学生”，实现鲁棒协同定位（第四章）。citeturn3search36turn16search1turn7search16turn7search0  

## 第一章最适合引用的核心文献清单

以下 20 篇组合覆盖你四条主线的“发展脉络锚点 + 方法代表 + 不足切入点”，最适合作为第一章核心引用池（建议将其作为参考文献的骨架，再按需要补充细分文献）：

[2], [9], [10], [13], [14], [18], [19], [20], [21], [22], [23], [24], [30], [31], [39], [40], [41], [42], [43], [46]。citeturn4search6turn5search0turn2search27turn5search22turn15search3turn19search0turn18search0turn18search14turn0search20turn19search5turn19search3turn19search22turn16search0turn16search1turn13search2turn15search2turn15search13turn15search4turn15search27turn6search16  

## 第一章按综述段落组织的引用建议

下面给出一种“可直接照搬”的第一章引用组织方式（示例段落划分可与你实际目录对应调整）。

**段落：研究背景与问题引出（LEO SoOP 可行性 + 城市复杂环境挑战）**  
建议引用：[22][23][19][10]。用 [22][23] 说明 SoOP/海量 LEO PNT 的研究热度与挑战框架，用 [19] 给出 Starlink PNT 的实验代表性成果，再用 [10] 引出城市环境下传播统计特性导致观测退化。citeturn19search5turn19search3turn18search0turn2search27  

**段落：方向一国内外进展（Starlink 下行解析与可观测量提取）**  
建议引用：[18][19][20][21]。写作结构：先用 [18] 说明 OFDM-like 结构与帧、盲获取；再用 [19] 作为系统性/实验性“综述式锚点”；最后用 [20][21] 承接到“时频/可预测元素”影响，过渡到你后续“鲁棒融合/动态权重”的必要性。citeturn19search0turn18search0turn18search14turn0search20  

**段落：方向二国内外进展（城市多径/NLOS 退化与经典定位局限）**  
建议引用：[9][2][10][14]。用 [9] 给出 NLOS 偏置问题与缓解谱系；用 [2] 指出经典 TDOA 基线在理想条件下有效但对大误差敏感；用 [10] 给出标准化城市信道统计；用 [14] 作为“地图/射线追踪/指纹 + 学习”增强路线代表。citeturn5search0turn4search6turn2search27turn15search3  

**段落：方向三国内外进展（UAV 机动采样与 Radio Map/REM 重构）**  
建议引用：[39][40][41][42][43][46]。写作结构：用 [39] 作为 UAV + Radio Map 学习的早期代表；用 [40][41] 说明深度学习从几何/稀疏测量生成地图；用 [42] 强调主动采样与在线更新；用 [43][46] 分别作为“方法谱系综述”和“近年补充综述”。citeturn13search2turn15search2turn15search13turn15search4turn15search27turn6search16  

**段落：方向四国内外进展（RL 与 KD 在定位鲁棒化中的应用）**  
建议引用：[31][30][32][34][35][37]。写作结构：用 [31] 把 RL 与定位性能指标（CRLB）直接绑定，突出“不是泛泛智能化”；用 [30] 提供 UAV+无线定位+RL 的落地例子；用 [32][34] 简要锚定 MDP/DQN 基础；用 [35][37] 承接“KD 用于掉点/协同鲁棒”。citeturn16search1turn16search0turn3search36turn3search26turn3search35turn7search16  

**适合写成对比关系的文献对（建议你第一章用“对比句式”写不足）**  
- “经典几何闭式解 vs 复杂环境鲁棒性”：[2] 对比 [14]。citeturn4search6turn15search3  
- “离线重构 vs 主动采样在线重构”：[40][41] 对比 [42]。citeturn15search2turn15search13turn15search4  
- “手工规则/固定权融合 vs 学习型调度/动态权重”：传统基线（可用 [2][26]）对比 RL 调度 [31]。citeturn4search6turn10search8turn16search1  

## 参考文献

[1] entity["people","Torrieri D J","passive geolocation author"]. Statistical Theory of Passive Location Systems[J]. IEEE Transactions on Aerospace and Electronic Systems, 1984, 20(2): 183-198. citeturn20search5  
[2] entity["people","Chan Y T","signal processing author"], entity["people","Ho K C","signal processing author"]. A simple and efficient estimator for hyperbolic location[J]. IEEE Transactions on Signal Processing, 1994, 42(8): 1905-1915. citeturn4search6  
[3] entity["people","Kay S M","estimation theory author"]. Fundamentals of Statistical Signal Processing: Estimation Theory[M]. Upper Saddle River, NJ, USA: Prentice Hall PTR, 1993. citeturn4search0  
[4] entity["people","Van Trees H L","statistical signal processing author"]. Detection, Estimation, and Modulation Theory, Part I: Detection, Estimation, and Linear Modulation Theory[M]. New York, NY, USA: Wiley, 1968. citeturn4search25  
[5] entity["people","Zekavat S A (Reza)","positioning handbook editor"], entity["people","Buehrer R Michael","positioning handbook editor"]. Handbook of Position Location: Theory, Practice and Advances[M]. Hoboken, NJ, USA: John Wiley & Sons, 2011. citeturn20search0turn20search11  
[6] entity["people","Weiss A J","geolocation author"]. Direct geolocation of wideband emitters based on delay and Doppler[J]. IEEE Transactions on Signal Processing, 2011, 59(6): 2513-2521. citeturn20search21turn20search6  
[7] entity["people","Amar A","signal processing author"], Weiss A J. Direct Position Determination: A Single-Step Emitter Localization Approach[C]//Classical and Modern Direction-of-Arrival Estimation. Oxford, UK: Academic Press, 2009. citeturn20search3turn20search22  
[8] entity["people","Wymeersch H","wireless localization author"], entity["people","Lien J","wireless localization author"], entity["people","Win M Z","wireless localization author"]. Cooperative localization in wireless networks[J]. Proceedings of the IEEE, 2009, 97(2): 427-450. citeturn16search3  
[9] entity["people","Güvenç İ","wireless localization author"], entity["people","Chong C C","wireless localization author"]. A Survey on TOA Based Wireless Localization and NLOS Mitigation Techniques[J]. IEEE Communications Surveys & Tutorials, 2009, 11(3): 107-124. citeturn5search0  
[10] entity["organization","3GPP","cellular standards org"]. TR 38.901: Study on channel model for frequencies from 0.5 to 100 GHz[R]. 3GPP Technical Report, v17.0.0 (建议以你论文实际采用版本为准). citeturn2search27  
[11] entity["people","O’Connor A C","rf localization author"], entity["people","Setlur P","rf localization author"], entity["people","Devroye N","rf localization author"]. Single-sensor RF emitter localization based on multipath exploitation[J]. (公开 PDF), 2013. citeturn5search22  
[12] entity["people","de Sousa M N","nlos localization author"], entity["people","Thomä R S","nlos localization author"]. Enhancement of Localization Systems in NLOS Urban Scenario with Multipath Ray Tracing Fingerprints and Machine Learning[J]. Sensors, 2018, 18(11): 4073. citeturn15search3  
[13] entity["organization","Federal Communications Commission","telecom regulator, us"]. DA 24-1193A1: SpaceX Gen2 Starlink applications and Ku/Ka-band processing round discussion[R]. 2024. citeturn21search7  
[14] Federal Communications Commission. DA 26-36A1: Authorization and frequency bands including 14.0–14.5 GHz (Earth-to-space) and related bands[R]. 2026. citeturn21search1  
[15] entity["company","SpaceX","aerospace company"]. SpaceX non-geostationary satellite system—Technical Attachment (IBFS filing, includes user downlink 10.7–12.7 GHz and user uplink 14.0–14.5 GHz)[R]. 2016. citeturn21search2  
[16] SpaceX. Ku-band uplink communications (14.0–14.5 GHz) referenced in FCC attachment for vehicle/spacecraft-related earth station operations[R]. 2021. citeturn21search3  
[17] entity["people","Neinavaie M","leo sop author"], Kassas Z M. Unveiling Starlink LEO Satellite OFDM-Like Signal Structure Enabling Precise Positioning[J]. IEEE Transactions on Aerospace and Electronic Systems, 2024, 60(2): 2486-2489. citeturn19search0  
[18] entity["people","Kozhaya S","leo pnt author"], entity["people","Saroufim J","leo pnt author"], Kassas Z M. Unveiling Starlink for PNT[J]. NAVIGATION: Journal of the Institute of Navigation, 2025, 72(1): navi.685. citeturn18search0  
[19] entity["people","Qin H L","starlink timing author"], et al. Timing Properties of the Starlink Ku-Band Downlink[EB/OL]. arXiv, 2025. citeturn18search14  
[20] Qin H L, et al. Pilots and other predictable elements in the Starlink downlink signal[EB/OL]. arXiv, 2026. citeturn0search20  
[21] entity["people","Stock W","leo pnt survey author"], entity["people","Schwarz R T","leo pnt survey author"], entity["people","Hofmann C A","leo pnt survey author"], entity["people","Knopp A","leo pnt survey author"]. Survey on Opportunistic PNT With Signals From LEO Communication Satellites[J]. IEEE Communications Surveys & Tutorials, 2025, 27(1): 77-107. citeturn19search5turn19search13  
[22] entity["people","Fan G","leo soop author"], et al. Toward Massive Satellite Signals of Opportunity Positioning: Challenges, Methods and Experiments[J]. Space: Science & Technology, 2024: 0191. citeturn19search3turn19search18  
[23] entity["people","Orabi M","leo doppler author"], entity["people","Khalife J","leo doppler author"], Kassas Z M. Opportunistic Navigation with Doppler Measurements from Iridium NEXT and Orbcomm LEO Satellites[J]. (公开 PDF/会议或期刊版本以你采用者为准), 2019. citeturn19search22  
[24] entity["people","Shahcheraghi S","leo doa author"], et al. Joint Doppler and Azimuth DOA Tracking for Positioning with Iridium NEXT LEO Signals of Opportunity[J]. (公开 PDF), 2023. citeturn19search2  
[25] entity["people","Patwari N","wsn localization author"], et al. Locating the Nodes: Cooperative Localization in Wireless Sensor Networks[J]. IEEE Signal Processing Magazine, 2005, 22(4): 54-68. citeturn10search8turn10search4  
[26] entity["people","Schmidt R O","music algorithm author"]. Multiple emitter location and signal parameter estimation[J]. IEEE Transactions on Antennas and Propagation, 1986, 34(3): 276-280. citeturn9search6  
[27] entity["people","Krim H","array processing author"], entity["people","Viberg M","array processing author"]. Two Decades of Array Signal Processing Research: The Parametric Approach[J]. IEEE Signal Processing Magazine, 1996, 13(4): 67-94. citeturn10search3turn10search35  
[28] entity["people","Harle R","pdr survey author"]. A Survey of Indoor Inertial Positioning Systems for Pedestrians[J]. IEEE Communications Surveys & Tutorials, 2013, 15(3): 1281-1293. citeturn9search4  
[29] entity["people","Wu G","uav rl localization author"]. UAV-Based Interference Source Localization: A Multimodal Q-Learning Approach[J]. IEEE Access, 2019, 7: 137982-137991. citeturn16search0  
[30] entity["people","Peng B","drl localization author"], et al. Decentralized Scheduling for Cooperative Localization With Deep Reinforcement Learning[J]. IEEE Transactions on Vehicular Technology, 2019, 68(5): 4295-4305. citeturn16search1  
[31] entity["people","Sutton R S","reinforcement learning author"], entity["people","Barto A G","reinforcement learning author"]. Reinforcement Learning: An Introduction (2nd ed.)[M]. Cambridge, MA, USA: MIT Press, 2018. citeturn3search36turn3search34  
[32] entity["people","Puterman M L","mdp author"]. Markov Decision Processes: Discrete Stochastic Dynamic Programming[M]. New York, NY, USA: Wiley, 1994. citeturn3search24  
[33] entity["people","Mnih V","deep rl author"], et al. Human-level control through deep reinforcement learning[J]. Nature, 2015, 518: 529-533. citeturn3search26turn3search23  
[34] entity["people","Hinton G","knowledge distillation author"], entity["people","Vinyals O","knowledge distillation author"], entity["people","Dean J","knowledge distillation author"]. Distilling the Knowledge in a Neural Network[EB/OL]. arXiv, 2015. citeturn3search35  
[35] entity["people","Wang L","kd survey author"], entity["people","Yoon K J","kd survey author"]. Knowledge Distillation and Student–Teacher Learning for Visual Intelligence: A Review and New Outlooks[EB/OL]. arXiv, 2020. citeturn7search19  
[36] entity["people","Gao Z","marl kd author"], et al. KnowRU: Knowledge Reuse via Knowledge Distillation in Multi-Agent Reinforcement Learning[J]. (开放获取论文), 2021. citeturn7search16  
[37] entity["people","Tseng W C","offline marl kd author"], et al. Offline Multi-Agent Reinforcement Learning with Knowledge Distillation[C]//NeurIPS. 2022. citeturn7search0turn7search20  
[38] entity["people","Chen J","uav radio map author"], entity["people","Yatnalli U","uav radio map author"], entity["people","Gesbert D","uav radio map author"]. Learning Radio Maps for UAV-aided Wireless Networks: A Segmented Regression Approach[C]//IEEE ICC. 2017. citeturn13search2turn13search12  
[39] entity["people","Levie R","radio map author"], entity["people","Yapar Ç","radio map author"], entity["people","Kutyniok G","radio map author"], entity["people","Caire G","radio map author"]. RadioUNet: Fast Radio Map Estimation with Convolutional Neural Networks[J]. IEEE Transactions on Wireless Communications, 2021, 20(6): 4001-4015. citeturn15search2  
[40] entity["people","Teganya Y","radio map author"], entity["people","Romero D","radio map author"]. Deep Completion Autoencoders for Radio Map Estimation[J]. IEEE Transactions on Wireless Communications, 2022, 21(3): 1710-1724. citeturn15search13turn15search8  
[41] entity["people","Shrestha R","uav radio map author"], Romero D, entity["people","Chepuri S P","uav radio map author"]. Spectrum Surveying: Active Radio Map Estimation with Autonomous UAVs[J]. IEEE Transactions on Wireless Communications, 2022, 22(1): 627- (页码以期刊终版为准). citeturn15search4turn15search12  
[42] Romero D, entity["people","Kim S-J","spectrum cartography author"]. Radio Map Estimation: A Data-Driven Approach to Spectrum Cartography[J]. IEEE Signal Processing Magazine, 2022. citeturn15search27turn15search8  
[43] entity["people","Zhao Y","rem author"], et al. Radio Environment Map enabled situation-aware cognitive algorithms for NR and WRAN systems[C]//IEEE DySPAN. 2006. citeturn2search8  
[44] entity["people","Wei B","rem author"], et al. On the Construction of Radio Environment Maps for Cognitive Radio Networks[J]. (相关来源页面可检索到该工作与 REM 构建主题关联), 2013. citeturn2search28  
[45] entity["people","Feng B","radio map survey author"], et al. A Recent Survey on Radio Map Estimation Methods for …[J]. Electronics, 2025, 14(8): 1564. citeturn6search16  
[46] entity["people","Chang C","localization bounds author"], entity["people","Sahai A","localization bounds author"]. Estimation Bounds for Localization[J]. EURASIP Journal on Applied Signal Processing, 2004. citeturn4search35  
[47] entity["people","Shi H","rssi crlb author"], et al. Cramer-Rao Bound Analysis of Quantized RSSI Based Localization[C]//IEEE ICPADS. 2005. citeturn4search7  
[48] entity["people","Pine J","tdoa fdoa geometry author"], entity["people","Cheney M","tdoa fdoa geometry author"]. The Geometry of Far-Field Passive Source Localization With TDOA/FDOA[J]. IEEE Transactions on Signal Processing, 2021. citeturn2search17