# 《基于低成本无人机平台的非法低轨卫星终端定位识别算法研究》第二章“系统模型与定位理论基础”深度文献调研报告

（你上传的论文 PDF 供你本地对照：sandbox:/mnt/data/b435cce5-d7ec-49ec-b641-4590fd0dd00d.pdf）

## 执行摘要
本调研围绕第二章“系统模型与定位理论基础”构建可直接落地的文献支撑链条：首先以无源/非合作辐射源定位的统计建模与两步法（参数估计→定位解算）为主线，补充直接定位（DPD）以解释“信号级端到端定位”趋势；其次针对城市复杂传播，引入3GPP/ITU的宏/微小区外场模型与延时扩展参数化，结合TOA/TDOA在NLOS偏差下的鲁棒化与理论边界；第三汇总Starlink/LEO Ku下行公开研究（波段占用、240 MHz信道、OFDM栅格、同步序列、帧周期与时钟稳定性），明确“官方未公开帧/导频细节，需以逆向测量与仿真替代”；第四给出RSSI/AoA/TOA/TDOA/PDR的经典观测方程、常用解算与误差指标，并以CRLB/FIM串接“理论极限—算法设计—实验验证”。同时补齐MDP/DQN与知识蒸馏的基础定义与核心公式，便于你在后续章节引入“智能搜索/自适应融合/轻量部署”。整体输出包含：分小节文献表、逐篇“可支撑公式/概念/引用优先级”说明、建议直接写入第二章的公式与引用挂点，以及写作建议。 citeturn0search0turn3search5turn6search0turn14search4turn2search0turn15search0turn5search5turn3search5

## 文献筛选边界与章节结构对齐
本报告遵循你的排除要求：不纳入“仅WiFi室内定位（且不涉及复杂城市/无人机平台）”、不纳入“纯视觉检测（无无线观测/Radio Map/定位）”、不纳入“与定位观测无关的一般路径规划”、不纳入“纯卫星通信链路优化（不涉及非合作感知/定位/排查）”。因此，保留的“室内相关”内容仅限于：**可迁移的统计建模/CRLB/NLOS方法论**（例如NLOS识别与TOA偏差理论），且会在文献注释中明确其外推边界。 citeturn17search5turn17search1turn3search5turn6search0

从你给出的第二章主题来看（系统模型→传播与信号结构→经典定位→MDP/DQN→性能评估），可将“法理链条”写成一条主线：  
**观测与状态（系统模型） → 传播扰动（城市多径/NLOS）+ 信号结构（Ku/OFDM帧） → 观测量（RSSI/AoA/TOA/TDOA/PDR） → 解算器（LS/WLS/ML/DPD、滤波） → 指标与理论下界（RMSE/CEP/CRLB） → 智能策略（MDP/DQN）与轻量化部署（KD）**。这一主线在IEEE SPM 2005专题中是典型组织方式，可直接复用其“测量—算法—误差源—性能界”写作框架。 citeturn0search0turn3search5turn19search0turn7search0

## 非合作式无源定位系统模型与统计定位理论
### 综述性总结（150–250字，建议放在第二章“系统模型/定位体制”开头）
非合作无源定位以“被动接收辐射源信号、未知发射时刻/波形细节、平台几何随时间变化”为典型特征，其理论主线通常采取两步法：先从多站观测估计中间参数（AoA/TOA/TDOA/FDOA或其统计量），再通过几何约束与最优化获得位置信息；该范式易工程实现、可模块化替换估计器，但在低SNR与城市多径/NLOS条件下会出现参数估计偏差与一致性破坏。为弥补两步法的“信息损失”，直接定位（DPD）将位置作为待估参数，在信号级联合利用多站观测进行ML型估计，理论上更接近最优，但代价是更强的模型依赖与更高的计算复杂度，并对多径/阵列校准误差敏感。经典统计理论进一步给出FIM/CRLB框架，用以统一刻画不同观测量下的可达精度，并指导无人机平台的布站几何与观测设计。**推荐在段末挂引**：[1],[2],[3],[4],[5],[6],[7]。 citeturn0search0turn3search5turn6search0turn7search0turn17search1turn10search26turn10search5

### 可直接支撑写作的核心文献与“抓手”
经典体制与误差源框架（必须引用）
- **[1]**给出网络辅助定位的统一建模视角（可作为你“系统模型 + 观测量分类”的总纲），适合引到：观测量类型、同步误差、NLOS偏差、两步法链路。 citeturn0search0  
- **[2]**强调“可用测量决定可达精度”的观点，适合用来引入：GDOP/几何构型、同步/带宽对精度的根本限制。 citeturn3search5  
- **[3]**提供“信号处理视角的网络定位综述”，可补强你在2.3节对“测量→估计器→解算器”的组织。 citeturn19search0  
- **[4]**是无源定位统计理论的里程碑，系统推导TDOA/AOA等体制的误差传播、性能分析（非常适合做2.5节CRLB/误差指标承上启下的“经典定义出处”）。 citeturn17search1  

两步法定位解算的代表性原始论文（必须引用）
- **[5] Chan–Ho（TDOA闭式/高效估计）**：适合写清TDOA双曲线定位的经典估计器与实现要点，便于在2.3节给出“工程可实现的基线算法”。 citeturn0search1  
- **[6] Foy（泰勒级数迭代）**：适合作为“迭代最小二乘/泰勒线性化”经典出处，可用于你在2.3节写“非线性方程组迭代求解”的统一模板。 citeturn16search19turn16search3  
- **[7] Schmidt（MUSIC）**：AoA/DOA超分辨的经典原始论文，可用于解释“阵列测角→几何交汇”的来向估计环节。 citeturn6search1turn6search37  

直接定位DPD代表性论文（建议引用）
- **[8],[9],[10]**（Weiss/Amar系）非常适合在2.1“系统模型与定位范式”中加入一小段“二步法 vs 直接定位”的对比，作为你后续“信号级识别/定位”算法（尤其面向非公开信号结构）的方法论支点。 citeturn10search0turn10search26turn10search5  

中文高质量综述（建议引用，用于中文论文叙述更顺）
- **[11]**“信号直接定位技术综述”从中文语境梳理“二步法→直接定位”的研究脉络，适合你在第二章开头写“研究背景+方法谱系”，并与[8]–[10]形成“中文综述+英文原始论文”的组合引用。 citeturn18search0  

## 城市复杂传播与Starlink/LEO Ku下行信号结构
### 复杂城市传播：多径、NLOS、时延扩展、衰落
#### 综述性总结（150–250字，建议放在第二章“传播/观测误差建模”小节开头）
城市环境的定位误差往往不是由算法形式决定，而是由传播机理造成的“不可忽略偏差”：遮挡与反射使直达径缺失或能量劣化，导致TOA/TDOA出现正偏差、AoA被强反射径牵引，且宽带信号会呈现显著的功率延迟谱（PDP）与时延扩展，从而引入测距分辨率与估计方差的根本限制。工程上通常以（i）大尺度：路径损耗+阴影衰落；（ii）小尺度：多径时延簇、RMS延时扩展、频率选择性衰落；（iii）LoS概率与街谷几何等统计量进行统一表征，其中3GPP/ITU的外场模型为论文提供可复现的参数来源。针对定位场景，主流思路是：识别/抑制NLOS链路、在优化中引入鲁棒代价或约束松弛，并用CRLB/SPEB评估“在给定NLOS比例/带宽/几何下”的理论可达精度与算法性能差距。**推荐挂引**：[12],[13],[14],[15],[16],[17],[18]。 citeturn2search0turn2search1turn11search0turn7search18turn17search5turn16search37turn17search1turn7search31

#### 关键支撑文献（按可写入公式/参数的价值排序）
- **[12] 3GPP TR 38.901（必须引用）**：提供UMa/UMi等场景下路径损耗、阴影衰落、延时扩展（DS）、角度扩展等统计参数，是你写“城市复杂环境信道参数化”的最强“可复现出处”。 citeturn2search0  
- **[13] ITU-R P.1411（必须引用）**：短距室外传播预测（含街道/障碍物影响）常作为城市场景建模的权威标准来源，适合与3GPP并列作为“标准级引用”。 citeturn2search1  
- **[14] Andersen–Rappaport–Yoshida（建议引用）**：经典传播测量与建模综述，适合用于“多径、衰落、延时扩展为何影响系统设计”的文字论证与定义引用。 citeturn11search0turn11search4  
- **[15] Sun 等（建议引用）**：给出LoS概率、路径损耗、阴影衰落的测量与建模方法，虽面向5G频段，但方法对Ku/上中频城市场景具有迁移价值。 citeturn7search6turn7search10  
- **[16] Guvenc–Chong（建议引用）**：TOA定位与NLOS缓解的高质量综述，适合写“为什么NLOS是TOA/TDOA的核心难点、常见缓解分类”。 citeturn17search5turn17search7  
- **[17] Zou 等（可选）**：给出基于凸优化/SDP的TOA-NLOS误差缓解思路，适合作为“近年补充：鲁棒优化路线”的代表。 citeturn16search37turn17search18  
- **[18] Amar–Weiss（可选）**：讨论DPD在模型误差（含多径、校准误差等）下的性能退化，适合把“城市多径对直接定位同样致命”写得更有文献支撑。 citeturn10search5  

### Starlink/LEO Ku下行链路信号结构与公开资料边界
#### 综述性总结（150–250字，建议放在第二章“LEO信号结构/可利用特征”小节开头）
Starlink Ku下行可被视作一种高功率、宽带、强多普勒的“机会信号”，其在PNT/监管排查/非法终端识别领域的研究热度源于：星座密度高、地面接收门槛低、可从同步结构中提取伪距/时间基准。然而，官方公开材料通常仅覆盖频段占用与干扰论证等监管信息，并不披露帧结构、导频/同步序列等细节；因此学术界主要通过实测逆向与可重复仿真平台揭示其OFDM栅格、240 MHz信道化、帧周期、同步序列与时钟特性，并进一步评估其用于伪距/载波/多普勒观测的可行性。对你的论文第二章而言，应明确区分：**（i）可直接引用的公开/官方频谱与信道化信息；（ii）需依赖公开论文逆向得到的帧/同步细节；（iii）仍未公开或代际可变的内容（需标注“未公开/需推断”，并以仿真或测量替代）**。**推荐挂引**：[19],[20],[21],[22],[23]。 citeturn13search3turn13search6turn14search4turn14search6turn13search2turn13search1turn14search23

#### “官方/公开资料”与“逆向论文”两类关键来源
- **[19] SpaceX 12 GHz干扰研究报告（必须引用，官方文件）**：明确Starlink Ku下行频段划分与“7个240 MHz信道、250 MHz间隔、10.95–12.7 GHz”等参数，适合作为你“频谱/信道化”数据的权威出处（可附引用即可形成“官方链接”）。 citeturn13search3  
- **[20] FCC公开材料（建议引用）**：在监管语境再次引用上述信道化假设，可用作“第三方监管复述”支撑。 citeturn13search6turn13search9  
- **[21] Humphreys 等：Starlink Ku下行结构（必须引用，代表性逆向论文）**：给出OFDM结构细节、同步序列等，并明确指出“波形细节不公开”，是你写“未公开→需逆向”的关键证据来源。 citeturn14search4turn14search7  
- **[22] Qin 等：时序性质（建议引用）**：聚焦帧定时稳定性与是否受GPS时标约束，适合你在2.5“可作为伪距观测的前提条件”处引用。 citeturn14search6turn13search19  
- **[23] Komodromos–Qin–Humphreys：信号仿真器（建议引用）**：提供“可复现仿真框架”，当你需要在论文中用仿真验证算法或推导CRLB时，这类平台是最合适的“仿真依据”。 citeturn13search2turn13search5  
- **[24] Neinavaie–Kassas：OFDM-like结构与盲接收（可选，近年补充）**：用于补充“盲接收/顺序GLRT获取多星”的研究路线，与你的“非法终端识别/信号级特征提取”叙述高度兼容。 citeturn13search0turn13search4  
- **[25] Kozhaya 等：Unveiling Starlink for PNT（可选，但很强）**：系统化呈现“Starlink用于PNT”的理论与实验链条，可作为你第二章末尾“相关信号可用于定位观测”的权威补充。 citeturn13search1turn13search22  
- **[26] Qin 等：Pilots与可预测元素（可选，最新补充）**：若你第二章要写“可利用导频/模板提升处理增益”，该文提供了非常直接的可引用结论（适合第三章/第四章也复用）。 citeturn14search23  

> **对“帧结构是否有官方规范”的结论写法建议**：  
> 可写为：“监管/官方公开材料公开了频段与信道化假设，但波形与帧级细节未公开；现有学术研究通过实测逆向给出OFDM栅格、同步序列与帧定时特性，本文据此建立可复现的信号模型。”并分别引用[19]/[20]与[21]/[22]/[23]。 citeturn13search3turn13search6turn14search4turn14search6turn13search2  

## 经典定位观测模型与解算：RSSI/AoA/TOA/TDOA/PDR
### 综述性总结（150–250字，建议放在第二章“定位观测模型”小节开头）
经典无线定位模型可按“观测量—几何约束—统计估计”展开：RSSI基于路径损耗模型将功率映射为距离，但对城市阴影衰落与遮挡高度敏感，常作为粗定位或与其他量融合；AoA依赖阵列/多天线测角形成射线交汇，受多径角度扩展与阵列校准误差影响；TOA/TDOA利用传播时延构造圆/双曲线约束，若缺乏严格同步或存在NLOS正偏差，易出现系统性定位漂移，因此工程上常采用WLS/ML、鲁棒代价或NLOS剔除来提高一致性。PDR则以惯性里程计提供短时连续性，通过步长/航向积分更新位置，误差随时间漂移，需依靠外部无线观测进行重校正。论文写作中建议明确：各观测量的数学模型、噪声统计假设、方程组求解方式（闭式/迭代/滤波）与适用条件，并在第二章末以CRLB/FIM统一比较“带宽、几何、NLOS比例”对精度的影响。**推荐挂引**：[4],[5],[6],[7],[27],[28],[29],[30]。 citeturn17search1turn0search1turn16search19turn6search1turn6search0turn17search5turn12search35turn7search19

### 建议你在第二章明确给出并“挂出处”的典型模型/算法
- **TDOA双曲线定位的两步法基线**：用[5]作为闭式/高效估计器代表；用[6]作为迭代线性化代表；用[4]给出无源定位误差传播的经典理论底座。 citeturn0search1turn16search19turn17search1  
- **AoA/DOA测角与超分辨**：用[7]作为MUSIC原始出处，强调阵列协方差特征分解与信号/噪声子空间分离。 citeturn6search1turn6search21  
- **测量噪声非同方差/距离相关噪声**：城市环境下TDOA测量误差常呈距离相关或受NLOS影响，可用[29]（距离相关噪声TDOA定位）为你后续“加权/鲁棒”铺垫。 citeturn16search1  
- **NLOS缓解的分类与代表方法**：可用[16]写综述分类，用[17]给出凸优化/SDP路线代表。 citeturn17search5turn16search37  
- **CRLB贯穿式比较**：RSS/TOA混合与TDOA/RSS混合的CRLB可用[31]（Patwari：RSS/TOA CRLB思路）与[32]（Catovic：TOA/RSS与TDOA/RSS CRB）作为可直接引用的推导来源。 citeturn7search19turn15search3turn15search23  

> PDR部分若你第二章仅需“模型定义+误差项”，可用惯导/PDR综述类权威教材补充；本次调研重点放在你明确要求的“无线观测量+统计定位理论”主链。

## MDP/DQN基础与知识蒸馏基础
### 马尔可夫决策过程与DQN
#### 综述性总结（150–250字，建议放在第二章“强化学习基础”小节开头）
将无人机平台的观测调度、权重自适应或搜索行为建模为MDP，可把“状态（平台位姿、观测质量/NLOS概率、先验不确定度）—动作（机动/指向/采样/融合权重）—奖励（定位误差下降、信息增益、资源消耗）”统一到序贯决策框架中。MDP的理论核心是Bellman最优性方程与价值函数/策略的等价表述，经典教材提供严谨定义与收敛性质；在高维连续状态下，DQN以深度网络近似Q函数，通过经验回放与目标网络稳定训练，使得在未知环境中可学习近似最优策略。对论文第二章而言，应聚焦：MDP五元组定义、折扣回报、Bellman方程与Q-learning更新；并说明DQN相比表格Q-learning的必要性（函数逼近）及其训练稳定化技巧。**推荐挂引**：[33],[34],[35],[36]。 citeturn15search0turn15search1turn4search0turn4search1

### 知识蒸馏
#### 综述性总结（150–250字，建议放在第二章“模型轻量化/部署”小节开头）
知识蒸馏（KD）面向“性能—算力/存储—时延”的工程矛盾：通过让轻量学生网络拟合教师网络的软输出分布（温度软化后的soft targets），学生不仅学习硬标签，还学习类别间相对概率所携带的“暗知识”，从而在小模型上逼近大模型性能。KD最初以“软标签交叉熵+温度”形式提出，随后发展出特征蒸馏（中间层hint）、关系蒸馏与任务特化蒸馏等分支，并形成较成熟的综述体系。对你的论文而言，KD在低成本无人机载算力条件下尤为关键：可将复杂的信号识别/定位网络作为教师，在保证精度的同时部署轻量学生模型用于实时端侧推断，并在实验中比较蒸馏前后模型的精度、推理时延与能耗。**推荐挂引**：[37],[38],[39]。 citeturn4search3turn5search2turn5search5

## 定位误差指标与CRLB：公式挂点与写作落地
### 综述性总结（150–250字，建议放在第二章“性能评估”小节开头）
定位性能评估应区分“实验误差指标”与“理论下界”。前者以RMSE/MAE、水平/三维误差、CEP等统计量呈现算法在特定数据集与场景下的平均表现；后者以Fisher信息矩阵（FIM）与CRLB刻画在给定观测模型、噪声统计与几何构型下任何无偏估计器可达的方差下界，从而将“带宽、SNR、同步误差、NLOS比例、观测几何”对精度的影响显式化。无源定位领域的经典统计理论为不同体制（AOA/TDOA/混合）推导了误差传播与下界形式，现代综述进一步给出SPEB等位置误差界的统一表达；混合观测（如TOA/RSS、TDOA/RSS）能够在部分同步或弱几何条件下提升信息量，其CRLB推导也为你在算法设计时选择“融合哪些观测、如何加权”提供直接理论依据。**推荐挂引**：[4],[30],[31],[32],[40]。 citeturn17search1turn7search0turn7search19turn15search3turn15search2

### 建议你在第二章中“明确写出并引用”的公式/定理清单
下面给出可直接写入第二章的“建议公式编号（你可按论文排版调整）—公式内容—应挂引用”的清单；其中“引用位置”按你第二章小节主题组织。

**系统模型与观测（建议放2.1）**
- **（建议式2-1）观测模型/参数模型**：  
  \[
  \mathbf{z}_k=\mathbf{h}(\mathbf{x}_k)+\mathbf{v}_k,\quad \mathbf{v}_k\sim \mathcal{N}(\mathbf{0},\mathbf{R}_k)
  \]  
  **用途**：统一承载RSSI/AoA/TOA/TDOA等观测与噪声假设；后续FIM推导直接复用。  
  **建议引用**：网络定位综述/系统建模框架 [1],[3]。 citeturn0search0turn19search0  

**城市多径/NLOS与时延扩展（建议放2.2.1）**
- **（建议式2-2）大尺度路径损耗与阴影衰落（模型级）**：引用3GPP/ITU给出LoS/NLoS路径损耗表达与参数来源（不要只给“经验n值”，要给标准出处）。  
  **建议引用**：[12],[13]。 citeturn2search0turn2search1  
- **（建议式2-3）RMS延时扩展定义（从PDP得到）**：给出\(\mu_\tau\)、\(\sigma_\tau\)由PDP加权矩计算的定义，并指出其决定频率选择性与测距分辨率。  
  **建议引用**：经典传播综述 [14]；或以3GPP参数化说明 [12]。 citeturn11search0turn2search0  

**Starlink/Ku OFDM信号结构（建议放2.2.2）**
- **（建议式2-4）OFDM时域信号表达与CP（用于引出同步/帧结构）**：可采用通用CP-OFDM表达式，并强调“Starlink具体同步序列/导频来自逆向”。  
  **建议引用（通用OFDM定义）**：3GPP TS 38.211作为“通用OFDM符号/CP概念”较权威 [41]；  
  **Starlink具体结构**：必须挂[21]，并在文字中说明“官方未公开波形细节”。 citeturn9search0turn14search4  
- **（建议式2-5）信道化/频段占用（文本或表格即可）**：写明“7×240 MHz、250 MHz spacing、10.95–12.7 GHz”等，并标记为官方公开。  
  **建议引用**：[19],[20]。 citeturn13search3turn13search6  

**RSSI/AoA/TOA/TDOA经典测量方程（建议放2.3）**
- **（建议式2-6）TOA测距、TDOA差分测距**：  
  \[
  r_i=c\tau_i,\quad \Delta r_{ij}=c(\tau_i-\tau_j)
  \]  
  **建议引用**：无源定位统计理论 [4] 与两步法/定位综述 [1],[27]。 citeturn17search1turn0search0turn6search0  
- **（建议式2-7）TDOA双曲线定位与解算器**：给出最小二乘/加权最小二乘目标函数，并以[5]（闭式/高效）或[6]（迭代）为算法出处。 citeturn0search1turn16search19  
- **（建议式2-8）AoA射线几何与DOA估计**：MUSIC谱峰对应DOA，引用[7]。 citeturn6search1turn6search37  

**MDP/DQN（建议放2.4.1）**
- **（建议式2-9）MDP五元组定义**：\(\mathcal{M}=\langle \mathcal{S},\mathcal{A},P,R,\gamma\rangle\)。  
  **建议引用**：[33]。 citeturn15search0  
- **（建议式2-10）Bellman最优性方程与Q-learning更新**：引用[34]（RL教材）与[35]（Q-learning原论文）。 citeturn15search1turn4search0  
- **（建议式2-11）DQN损失函数/目标网络思想**：引用[36]（DQN）。 citeturn4search1turn4search5  

**知识蒸馏（建议放2.4.2或2.4.3）**
- **（建议式2-12）温度软化softmax与KD损失（KL/交叉熵形式）**：引用[37]作为原始出处，并可用[39]作为综述补充分类。 citeturn4search3turn5search5  

**CRLB/FIM（建议放2.5）**
- **（建议式2-13）CRLB定理（矩阵形式）**：  
  \[
  \mathrm{Cov}(\hat{\boldsymbol\theta})\succeq \mathbf{J}^{-1}(\boldsymbol\theta)
  \]
  其中\(\mathbf{J}\)为FIM。  
  **建议引用**：无源定位统计理论与CRLB推导 [4]；混合观测CRLB [32]；（若写SPEB/位置误差界）可引[30]/[31]。 citeturn17search1turn15search3turn7search0turn7search19  

## 第二章小节—文献映射总表
> 字段说明：优先级=必须/建议/可选；性质=经典/近年（近年一般指2018年以来，或对你研究问题有关键增量）。

| 第二章建议小节 | 文献编号 | 作者（et al.） | 年份 | 题目 | 来源/DOI（或标准编号） | 主要支撑内容（可直接写入） | 优先级 | 性质 |
|---|---:|---|---:|---|---|---|---|---|
| 2.1 系统模型与无源定位体制 | [1] | Sayed | 2005 | Network-based wireless location: challenges… | IEEE SPM, 22(4) | 系统建模框架、误差源分类、测量量综述 | 必须 | 经典 |
| 2.1 系统模型与无源定位体制 | [2] | Gustafsson | 2005 | Mobile positioning… fundamental limitations… | IEEE SPM, 22(4) | 可达精度与限制、GDOP/测量约束 | 必须 | 经典 |
| 2.1 系统模型与无源定位体制 | [3] | Sun | 2005 | Signal processing techniques in network-aided positioning | IEEE SPM, 22(4), DOI:10.1109/MSP.2005.1458273 | “测量→估计→解算→评估”的组织模板 | 建议 | 经典 |
| 2.1 系统模型与无源定位体制 | [4] | Torrieri | 1984 | Statistical Theory of Passive Location Systems | IEEE TAES, DOI:10.1109/TAES.1984.310439 | 无源定位统计理论、误差传播、性能分析 | 必须 | 经典 |
| 2.3 TDOA定位模型 | [5] | Chan | 1994 | A simple and efficient estimator for hyperbolic location | IEEE TSP | TDOA双曲线定位高效估计器 | 必须 | 经典 |
| 2.3 迭代解算 | [6] | Foy | 1976 | Position-Location Solutions by Taylor-Series Estimation | IEEE TAES, DOI:10.1109/TAES.1976.308294 | 泰勒线性化迭代LS模板 | 必须 | 经典 |
| 2.3 AoA/阵列测角 | [7] | Schmidt | 1986 | Multiple emitter location and signal parameter estimation | IEEE TAP, DOI:10.1109/TAP.1986.1143830 | MUSIC/DOA超分辨原始出处 | 必须 | 经典 |
| 2.1 两步法 vs 直接定位 | [8] | Weiss | 2004 | Direct position determination of narrowband RF transmitters | IEEE SPL, DOI:10.1109/LSP.2004.826501 | DPD概念/ML直接定位范式 | 建议 | 经典 |
| 2.1 两步法 vs 直接定位 | [9] | Weiss | 2005 | Direct Position Determination of Multiple Radio Signals | EURASIP JASP | 多源DPD框架 | 可选 | 经典 |
| 2.2.1 多径对模型误差影响 | [10] | Amar | 2006 | Direct position determination in the presence of model errors | Digital Signal Processing | 多径/校准误差对DPD影响 | 可选 | 经典 |
| 2.1 中文综述引入 | [11] | 吴癸周 | 2020 | 信号直接定位技术综述 | 雷达学报（DOI:10.12000/JR20040） | 中文语境梳理DPD脉络 | 建议 | 近年 |
| 2.2.1 城市信道参数化 | [12] | 3GPP | — | TR 38.901: Study on channel model… | 3GPP TR 38.901 | UMa/UMi路径损耗、DS等统计参数 | 必须 | 经典 |
| 2.2.1 室外短距传播标准 | [13] | ITU-R | — | Recommendation P.1411 | ITU-R P.1411 | 室外短距传播/街道场景预测方法 | 必须 | 经典 |
| 2.2.1 传播综述定义 | [14] | Andersen | 1995 | Propagation measurements and models… | IEEE Comm Mag, DOI:10.1109/35.339880 | 多径/衰落/延时扩展的传播解释 | 建议 | 经典 |
| 2.2.1 LoS概率/阴影衰落 | [15] | Sun | 2015 | Path Loss, Shadow Fading, and LOS Probability Models… | IEEE GLOCOMW, DOI:10.1109/GLOCOMW.2015.7414036 | LoS概率/路径损耗建模套路 | 建议 | 经典 |
| 2.2.1 NLOS缓解综述 | [16] | Guvenc | 2009 | A Survey on TOA Based Wireless Localization and NLOS Mitigation Techniques | IEEE ComST, DOI:10.1109/SURV.2009.090308 | NLOS识别/缓解分类、挑战 | 建议 | 经典 |
| 2.2.1 鲁棒NLOS（凸优化） | [17] | Zou | 2020 | An Efficient NLOS Errors Mitigation Algorithm for TOA-Based Localization | Sensors (MDPI) | SDP/凸松弛范式（可作近年补充） | 可选 | 近年 |
| 2.2.2 Ku频段与信道化（官方） | [19] | SpaceX | 2022 | Analysis of the Effect of Terrestrial… (12 GHz Interference Study) | SpaceX public file | 10.95–12.7 GHz；7×240 MHz信道等 | 必须 | 近年 |
| 2.2.2 监管复述（官方） | [20] | FCC | 2023 | Fact Sheet / Federal Register相关文件 | FCC公开PDF | 对[19]信道化假设的监管引用 | 建议 | 近年 |
| 2.2.2 Starlink帧/同步（逆向） | [21] | Humphreys | 2023 | Signal Structure of the Starlink Ku-Band Downlink | IEEE TAES, DOI:10.1109/TAES.2023.3268610 | OFDM栅格、同步序列、帧结构；并声明“未公开” | 必须 | 近年 |
| 2.2.2 Starlink定时特性 | [22] | Qin | 2025 | Timing Properties of the Starlink Ku-Band Downlink | arXiv:2501.05302 | 帧定时稳定性、时标约束讨论 | 建议 | 近年 |
| 2.2.2 Starlink仿真平台 | [23] | Komodromos | 2023 | Signal Simulator for Starlink Ku-Band Downlink | ION GNSS+ | 可复现信号仿真/检波与极限 | 建议 | 近年 |
| 2.2.2 盲接收/结构揭示 | [24] | Neinavaie | 2024 | Unveiling Starlink LEO Satellite OFDM-Like Signal Structure… | IEEE TAES, DOI:10.1109/TAES.2023.3265951 | OFDM-like RS、帧周期估计、GLRT | 可选 | 近年 |
| 2.2.2 PNT系统化描述 | [25] | Kozhaya | 2025 | Unveiling Starlink for PNT | NAVIGATION, DOI:10.33012/navi.685 | 理论+实验完整链条 | 可选 | 近年 |
| 2.2.2 导频/模板可预测元素 | [26] | Qin | 2026 | Pilots and Other Predictable Elements… | arXiv:2602.02627 | 导频/模板→处理增益（利于后续章节） | 可选 | 近年 |
| 2.3 定位方法综述总览 | [27] | Gezici | 2008 | A Survey on Wireless Position Estimation | WPC, DOI:10.1007/s11277-007-9375-z | RSS/AoA/TOA/TDOA统一综述 | 建议 | 经典 |
| 2.3 距离相关噪声TDOA | [29] | Huang | 2015 | TDOA-Based Source Localization With Distance-Dependent Noises | IEEE TWC, DOI:10.1109/TWC.2014.2351798 | 加权/噪声建模与理论分析 | 建议 | 经典 |
| 2.5 CRLB与协作定位框架 | [30] | Wymeersch | 2009 | Cooperative Localization in Wireless Networks | Proc. IEEE, DOI:10.1109/JPROC.2008.2008853 | FIM/CRLB、协作定位统一框架 | 必须 | 经典 |
| 2.5 CRLB（RSS/TOA） | [31] | Patwari | 2002 | Relative Location Estimation in Wireless Sensor Networks | IEEE TSP | RSS/TOA测量+CRLB分析框架 | 建议 | 经典 |
| 2.5 混合观测CRLB | [32] | Catovic | 2004 | CRB of Hybrid TOA/RSS and TDOA/RSS Schemes | IEEE ComL, DOI:10.1109/LCOMM.2004.835319 | 混合观测理论增益 | 建议 | 经典 |
| 2.4.1 MDP基础 | [33] | Puterman | 1994 | Markov Decision Processes… | Wiley, DOI:10.1002/9780470316887 | MDP严谨定义与理论性质 | 建议 | 经典 |
| 2.4.1 RL教材 | [34] | Sutton | 2018 | Reinforcement Learning: An Introduction | MIT Press（官方页） | Bellman/Q-learning/策略梯度等 | 必须 | 经典 |
| 2.4.1 Q-learning原始论文 | [35] | Watkins | 1992 | Q-learning | Machine Learning, DOI:10.1007/BF00992698 | Q-learning收敛与更新式出处 | 必须 | 经典 |
| 2.4.1 DQN原始论文 | [36] | Mnih | 2015 | Human-level control through deep RL | Nature, DOI:10.1038/nature14236 | DQN核心结构与训练稳定化 | 必须 | 经典 |
| 2.4.2 KD原始论文 | [37] | Hinton | 2015 | Distilling the Knowledge in a Neural Network | arXiv:1503.02531 | 温度softmax与软标签蒸馏 | 必须 | 经典 |
| 2.4.2 KD起源（模型压缩） | [38] | Bucilă | 2006 | Model Compression | KDD, DOI:10.1145/1150402.1150464 | 模型压缩早期代表 | 建议 | 经典 |
| 2.4.2 KD综述 | [39] | Gou | 2021 | Knowledge Distillation: A Survey | IJCV, DOI:10.1007/s11263-021-01453-z | KD分类与方法谱系 | 建议 | 近年 |
| 2.2.2 通用OFDM结构定义 | [41] | 3GPP | — | TS 38.211: NR Physical channels and modulation | ETSI PDF（3GPP TS 38.211） | CP-OFDM符号/CP等标准表述 | 可选 | 经典 |

## Mermaid：章节—文献映射关系图
```mermaid
graph TD
  A[第二章 系统模型与定位理论基础] --> B[2.1 非合作无源定位系统模型]
  A --> C[2.2 传播与信号结构]
  A --> D[2.3 经典定位观测模型与解算]
  A --> E[2.4 MDP/DQN 与知识蒸馏]
  A --> F[2.5 误差指标与CRLB]

  B --> B1[[1][2][3][4][5][6][7][8][9][10][11]]
  C --> C1[[12][13][14][15][16][17]]
  C --> C2[[19][20][21][22][23][24][25][26][41]]
  D --> D1[[4][5][6][7][27][29]]
  E --> E1[[33][34][35][36][37][38][39]]
  F --> F1[[4][30][31][32]]
```

## 下一步写作建议
第一，建议你在第二章开头用“二步法 vs 直接定位（DPD）”作为方法谱系主线，用[1],[4]给出无源定位统计框架，用[8],[9]点到信号级直接定位的优势与代价，再自然过渡到你后续章节的“非法终端识别+定位”的算法定位。 citeturn0search0turn17search1turn10search0turn10search26  

第二，城市传播建模建议采用“标准参数化 + 场景化补充”的写法：主参数出自3GPP/ITU（便于复现）[12],[13]，辅以经典传播综述解释物理意义[14]，并在定位角度用NLOS综述串联“偏差—缓解—边界”[16]。 citeturn2search0turn2search1turn11search0turn17search5  

第三，Starlink信号结构部分务必写清“公开边界”：频段与信道化用官方材料[19],[20]，帧/同步/导频等细节明确“未公开→依赖逆向测量”，并把[21]作为核心依据；若你第三章/第四章要做帧检测或模板相关，可提前埋下[23],[26]作为“可复现仿真/可预测元素”的技术支点。 citeturn13search3turn13search6turn14search4turn13search2turn14search23  

第四，第二章2.3建议采用“观测方程—噪声模型—解算器—适用条件”四段式，TDOA用[5]/[6]给基线算法，AoA用[7]给测角原始出处，并与2.5节用CRLB串起来：混合观测增益可引用[32]，统一框架引用[30]。 citeturn0search1turn16search19turn6search1turn15search3turn7search0  

第五，MDP/DQN与KD在第二章不宜写成“算法细节堆砌”，而应以“为什么需要（决策/轻量部署）—核心定义与损失函数—与定位任务的对应关系”为主：MDP定义引用[33]，Q-learning/DQN引用[35]/[36]，KD引用[37]并用[39]补充分类即可。 citeturn15search0turn4search0turn4search1turn4search3turn5search5  

## 参考文献（IEEE风格）
[1] A. H. Sayed, A. Tarighat, and N. Khajehnouri, “Network-based wireless location: Challenges faced in developing techniques for accurate wireless location information,” *IEEE Signal Processing Magazine*, vol. 22, no. 4, pp. 24–40, 2005. citeturn0search0  
[2] F. Gustafsson and F. Gunnarsson, “Mobile positioning using wireless networks: Possibilities and fundamental limitations based on available wireless network measurements,” *IEEE Signal Processing Magazine*, vol. 22, no. 4, pp. 41–53, 2005. citeturn3search5  
[3] G. Sun, J. Chen, W. Guo, and K. J. R. Liu, “Signal processing techniques in network-aided positioning: A survey of state-of-the-art positioning designs,” *IEEE Signal Processing Magazine*, vol. 22, no. 4, pp. 12–23, 2005, doi: 10.1109/MSP.2005.1458273. citeturn19search1turn19search0  
[4] D. J. Torrieri, “Statistical theory of passive location systems,” *IEEE Transactions on Aerospace and Electronic Systems*, vol. AES-20, no. 2, pp. 183–198, 1984, doi: 10.1109/TAES.1984.310439. citeturn17search1  
[5] Y. T. Chan and K. C. Ho, “A simple and efficient estimator for hyperbolic location,” *IEEE Transactions on Signal Processing*, 1994. citeturn0search1  
[6] W. H. Foy, “Position-location solutions by Taylor-series estimation,” *IEEE Transactions on Aerospace and Electronic Systems*, vol. AES-12, no. 2, pp. 187–194, 1976, doi: 10.1109/TAES.1976.308294. citeturn16search19turn16search3  
[7] R. O. Schmidt, “Multiple emitter location and signal parameter estimation,” *IEEE Transactions on Antennas and Propagation*, vol. 34, no. 3, pp. 276–280, 1986, doi: 10.1109/TAP.1986.1143830. citeturn6search1turn6search37  
[8] A. J. Weiss, “Direct position determination of narrowband radio frequency transmitters,” *IEEE Signal Processing Letters*, vol. 11, no. 5, pp. 513–516, 2004, doi: 10.1109/LSP.2004.826501. citeturn10search18turn10search15  
[9] A. J. Weiss and A. Amar, “Direct position determination of multiple radio signals,” *EURASIP Journal on Applied Signal Processing*, 2005. citeturn10search26  
[10] A. Amar and A. J. Weiss, “Direct position determination in the presence of model errors,” *Digital Signal Processing*, 2006. citeturn10search5  
[11] 吴癸周等, “信号直接定位技术综述,” *雷达学报*, 2020, doi: 10.12000/JR20040. citeturn18search0  
[12] 3GPP, “TR 38.901: Study on channel model for frequencies from 0.5 to 100 GHz.” citeturn2search0  
[13] ITU-R, “Recommendation ITU-R P.1411: Propagation data and prediction methods for the planning of short-range outdoor radiocommunication systems and radio local area networks.” citeturn2search1  
[14] J. B. Andersen, T. S. Rappaport, and S. Yoshida, “Propagation measurements and models for wireless communications channels,” *IEEE Communications Magazine*, vol. 33, no. 1, pp. 42–49, 1995, doi: 10.1109/35.339880. citeturn11search0turn11search4  
[15] S. Sun et al., “Path loss, shadow fading, and line-of-sight probability models for 5G urban macro-cellular scenarios,” in *IEEE GLOBECOM Workshops*, 2015, doi: 10.1109/GLOCOMW.2015.7414036. citeturn7search6turn7search10  
[16] I. Guvenc and C.-C. Chong, “A survey on TOA based wireless localization and NLOS mitigation techniques,” *IEEE Communications Surveys & Tutorials*, vol. 11, no. 3, pp. 107–124, 2009, doi: 10.1109/SURV.2009.090308. citeturn17search7turn17search5  
[17] Y. Zou et al., “An efficient NLOS errors mitigation algorithm for TOA-based localization,” *Sensors*, 2020. citeturn16search37turn17search18  
[18] （可按需补充）多径/模型误差对直接定位影响等，见[10]。 citeturn10search5  
[19] SpaceX, “Analysis of the Effect of Terrestrial … (12GHz Interference Study),” 2022. citeturn13search3  
[20] FCC, “Fact Sheet / Federal Register相关文件（引用SpaceX 7×240 MHz信道假设）,” 2023. citeturn13search6turn13search9  
[21] T. E. Humphreys et al., “Signal structure of the Starlink Ku-band downlink,” *IEEE Transactions on Aerospace and Electronic Systems*, vol. 59, pp. 6016–6030, 2023, doi: 10.1109/TAES.2023.3268610. citeturn14search4turn14search7  
[22] W. Qin et al., “Timing properties of the Starlink Ku-band downlink,” arXiv:2501.05302, 2025. citeturn14search6turn13search19  
[23] Z. M. Komodromos, W. Qin, and T. E. Humphreys, “Signal simulator for Starlink Ku-band downlink,” in *Proceedings of ION GNSS+*, 2023. citeturn13search2turn13search5  
[24] M. Neinavaie and Z. M. Kassas, “Unveiling Starlink LEO satellite OFDM-like signal structure enabling precise positioning,” *IEEE Transactions on Aerospace and Electronic Systems*, 2024, doi: 10.1109/TAES.2023.3265951. citeturn13search0turn13search4  
[25] S. Kozhaya, J. Saroufim, and Z. M. Kassas, “Unveiling Starlink for PNT,” *NAVIGATION*, vol. 72, no. 1, 2025, doi: 10.33012/navi.685. citeturn13search1turn13search22  
[26] W. Qin et al., “Pilots and other predictable elements of the Starlink Ku-band downlink,” arXiv:2602.02627, 2026. citeturn14search23  
[27] S. Gezici, “A survey on wireless position estimation,” *Wireless Personal Communications*, vol. 44, pp. 263–282, 2008, doi: 10.1007/s11277-007-9375-z. citeturn6search0  
[28] N. Patwari et al., “Locating the nodes: Cooperative localization in wireless sensor networks,” *IEEE Signal Processing Magazine*, vol. 22, no. 4, pp. 54–69, 2005, doi: 10.1109/MSP.2005.1458287. citeturn6search3turn7search24  
[29] B. Huang, L. Xie, and Z. Yang, “TDOA-based source localization with distance-dependent noises,” *IEEE Transactions on Wireless Communications*, vol. 14, no. 1, pp. 468–480, 2015, doi: 10.1109/TWC.2014.2351798. citeturn16search1  
[30] H. Wymeersch, J. Lien, and M. Z. Win, “Cooperative localization in wireless networks,” *Proceedings of the IEEE*, vol. 97, no. 2, pp. 427–450, 2009, doi: 10.1109/JPROC.2008.2008853. citeturn7search0  
[31] N. Patwari et al., “Relative location estimation in wireless sensor networks,” *IEEE Transactions on Signal Processing*, 2002. citeturn7search19  
[32] A. Catovic and Z. Sahinoglu, “The Cramer-Rao bounds of hybrid TOA/RSS and TDOA/RSS location estimation schemes,” *IEEE Communications Letters*, vol. 8, no. 10, pp. 626–628, 2004, doi: 10.1109/LCOMM.2004.835319. citeturn15search23turn15search3  
[33] M. L. Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley, 1994, doi: 10.1002/9780470316887. citeturn15search0  
[34] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018. citeturn15search25turn15search9  
[35] C. J. C. H. Watkins and P. Dayan, “Q-learning,” *Machine Learning*, vol. 8, pp. 279–292, 1992, doi: 10.1007/BF00992698. citeturn4search0  
[36] V. Mnih et al., “Human-level control through deep reinforcement learning,” *Nature*, vol. 518, pp. 529–533, 2015, doi: 10.1038/nature14236. citeturn4search1turn4search5  
[37] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv:1503.02531, 2015. citeturn4search3turn4search7  
[38] C. Bucilă, R. Caruana, and A. Niculescu-Mizil, “Model compression,” in *Proc. ACM SIGKDD*, 2006, doi: 10.1145/1150402.1150464. citeturn5search2turn5search14  
[39] J. Gou et al., “Knowledge distillation: A survey,” *International Journal of Computer Vision*, 2021, doi: 10.1007/s11263-021-01453-z. citeturn5search5turn5search1  
[40] M. Laaraiedh, S. Avrillon, and B. Uguen, “Cramer–Rao lower bounds for nonhybrid and hybrid localisation techniques in wireless networks,” *Transactions on Emerging Telecommunications Technologies*, 2012, doi: 10.1002/ett.1530. citeturn15search2turn15search14  
[41] 3GPP, “TS 38.211: NR; Physical channels and modulation,” ETSI PDF. citeturn9search0