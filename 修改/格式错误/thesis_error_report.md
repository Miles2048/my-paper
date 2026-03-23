# 大论文低级错误审查报告

> [!IMPORTANT]
> 本报告仅列出低级错误（错别字、病句、重复句、格式不统一、引用错误等），**不涉及任何内容修改建议**。

---

## 一、重复段落 / 重复句

### 1.1 第1章 §1.2 整段重复（严重）

[Chapter_01_Introduction.tex](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex) 中，第 41–49 行的内容与第 25–33 行**几乎完全重复**。

| 行号 | 内容摘要 |
|------|----------|
| L25–33 | §1.2.1 定位技术研究现状，从"覆盖能力方面……"到"……仍具有较强的现实意义" |
| L41–49 | §1.2.2 定位算法研究现状，从"覆盖能力方面……"到"……仍具有较强的现实意义" |

两处段落文字几乎一字不差，疑似复制粘贴遗留。L41 从一句残句"覆盖能力方面存在明显不足……"开始，没有上下文衔接，明显是拼接错误。

---

## 二、章节编号 / 标题风格不一致

### 2.1 正文章节号与文件名 / 注释严重不匹配

[main.tex](file:///Users/miles/Desktop/%E5%A4%A7%E8%AE%BA%E6%96%87/BUPTGraduateThesisLatexTemplate/main.tex) 中：
- L83 注释：`% 原第三章（单目标射频指纹定位）已被精简整合，故移除独立章节`
- L84：`\include{Chapter/Chapter_04_Fusion} % 第三章`
- L85：`\include{Chapter/Chapter_05_Multi_Target} % 第四章`
- L86：`\include{Chapter/Chapter_06_Conclusion} % 第五章`

**文件名是 `Chapter_04`、`Chapter_05`、`Chapter_06`，但注释和正文中实际引用为"第三章""第四章""第五章"。** 例如：

| 位置 | 引用方式 | 问题 |
|------|----------|------|
| [Chapter_02_Fundamentals.tex:L60](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_02_Fundamentals.tex#L60) | "在第三章中……在第四章中……" | ✓ 与 main.tex 注释一致 |
| [Chapter_02_Fundamentals.tex:L128](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_02_Fundamentals.tex#L128) | "在第三章中……在第四章中……" | ✓ 一致 |
| [Chapter_04_Fusion.tex:L5](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L5) | `\label{sec:3_1}` 等所有 label 以 `3_` 开头 | ✓ 但文件名是 `Chapter_04` |
| [Chapter_05_Multi_Target.tex:L5](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex#L5) | `\label{sec:4_1}` 等所有 label 以 `4_` 开头 | ✓ 但文件名是 `Chapter_05` |
| [Chapter_06_Conclusion.tex:L5](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_06_Conclusion.tex#L5) | `\label{sec:5_1}` 等所有 label 以 `5_` 开头 | ✓ 但文件名是 `Chapter_06` |

> [!WARNING]
> 文件名和 label 编号互相矛盾：文件名说第4/5/6章，正文/label说第3/4/5章。编译后章号由 LaTeX 自动编号（第3/4/5章），所以功能上无误，但**文件名极易误导维护**。需要确认是否需要重命名文件或调整 label 风格使其一致。

### 2.2 第1章 §1.4 "论文结构安排"中的描述

[Chapter_01_Introduction.tex:L72](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex#L72)：
> 本文共分为**五章**

实际编译产出也是五章（第1–5章），这一点是**正确**的（旧的第3章已被移除）。

### 2.3 第1章 §1.4 第4章描述语病

[Chapter_01_Introduction.tex:L77](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex#L77)：
> 第 4 章：基于信号成像与知识蒸馏方法，研究掉点情况下的 **利用Radio Map** 定位算法与视觉驱动的多目标识别定位算法。

- "利用Radio Map"前面莫名多了一个空格+缺少逗号/连词，读起来不通顺。
- "利用"与前面"研究掉点情况下的"拼接不自然，属于病句。

### 2.4 `sec:1_3` 之后直接跳到 `sec:1_4`，缺少 `sec:1_4` 对应的"主要创新点"小节

[Chapter_01_Introduction.tex:L60](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex#L60)：`\section{本文主要研究内容}` 的 label 为 `sec:1_4`。
[Chapter_01_Introduction.tex:L70](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex#L70)：`\section{论文结构安排}` 的 label 为 `sec:1_6`。

**label 编号从 `sec:1_4` 直接跳到 `sec:1_6`**，中间缺少 `sec:1_5`（可能是"主要创新点"小节被移除但 label 未更新）。

---

## 三、图表引用 / 图片路径问题

### 3.1 Chapter_04_Fusion.tex 图片路径格式不统一

| 行号 | 路径 | 问题 |
|------|------|------|
| L18 | `ch3/1.png` | 无 `figures/` 前缀 |
| L66 | `figures/ch2/1.jpeg` (Chapter_02) | 有 `figures/` 前缀 |

[main.tex](file:///Users/miles/Desktop/%E5%A4%A7%E8%AE%BA%E6%96%87/BUPTGraduateThesisLatexTemplate/main.tex) L28 已设置 `\graphicspath{{figures/}}`，所以 `ch3/1.png` 实际路径为 `figures/ch3/1.png`，功能上正确，但与第2章中 `figures/ch2/1.jpeg` 的写法**风格不一致**。

**Chapter_04_Fusion.tex 全部使用 `ch3/xxx` 格式（无 `figures/` 前缀），Chapter_02 全部使用 `figures/ch2/xxx` 格式（带 `figures/` 前缀）。** 两章风格不统一。

### 3.2 Chapter_05_Multi_Target.tex 中图片与文字描述可能不对应

[Chapter_05_Multi_Target.tex:L650](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex#L650)：

```latex
\includegraphics[width=0.92\textwidth]{ch4/8_不同掉点比例下的定位误差对比图.png}
\caption{教师—学生网络架构示意图}
```

**图片文件名为"不同掉点比例下的定位误差对比图"，但 caption 写的是"教师—学生网络架构示意图"。** 文件名和 caption 严重不匹配，需核实是否放错了图。

---

## 四、正文数据与表格 / 引用不一致

### 4.1 第3章正文提到"具体参数如下表3-3所示"但无对应表号

[Chapter_04_Fusion.tex:L744](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L744)：
> 上述方法覆盖了单源定位、模型驱动融合以及强化学习驱动融合三类典型技术路线……具体参数如下**表3-3**所示。

正文中使用硬编码"表3-3"而非 `\ref{}` 引用。且该位置附近只有 `\ref{tab:ch3_training_parameters}`（表3-2）和 `\ref{tab:ch3_overall_performance}`，**没有"表3-3"这个硬编码编号对应的表格**。应改用 `\ref{}` 交叉引用。

---

## 五、公式 / 变量 / 符号不一致

### 5.1 DQN 损失函数中下标不统一

[Chapter_04_Fusion.tex:L100](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L100)：
```latex
y_i-Q(s,a;\theta)
```
此处 target 用 $y_i$，但前面 L93 和 L104 中均使用 $y_i$，而公式定义中 L381–L383 （DQN基础章）使用 $y_t$。**同一个量在不同位置用了 $y_i$ 和 $y_t$ 两种下标，不统一。**

### 5.2 经验回放池符号不统一

| 位置 | 符号 |
|------|------|
| [Chapter_04_Fusion.tex:L97](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L97) | $D$ |
| [Chapter_04_Fusion.tex:L643](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L643) | $\mathcal{D}$ |
| [Chapter_04_Fusion.tex:L669](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L669)（伪代码） | $\mathcal{D}$（训练数据集）和 $\mathcal{B}$（经验回放池） |
| [Chapter_02_Fundamentals.tex:L398](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_02_Fundamentals.tex#L398) | $\mathcal{D}$ |

§3.1.3 中用 $D$（无花体），其余处用 $\mathcal{D}$ 或 $\mathcal{B}$。

### 5.3 建筑集合符号 $\mathcal{B}$ 与候选区域 $\mathcal{B}_m$ 冲突

- [Chapter_05_Multi_Target.tex:L249](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex#L249)：$\mathcal{B}=\{B_1,B_2,\ldots,B_M\}$ 表示建筑物集合
- [Chapter_05_Multi_Target.tex:L216](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex#L216)：$\mathcal{B}_m$ 表示第 $m$ 个检测到的目标区域

同一章中 $\mathcal{B}$ 既表示建筑物集合又作为目标区域的下标基符号使用，存在歧义。

---

## 六、缩写使用规范问题

### 6.1 首次使用缩写时未给全称

| 缩写 | 首次出现位置 | 问题 |
|------|------------|------|
| RSSI | [Chapter_01_Introduction.tex:L9](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex#L9) | 第1章正文首次出现即直接用 RSSI 缩写，未给全称。全称在摘要中出现过但正文应独立 |
| AoA | [Chapter_01_Introduction.tex:L9](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex#L9) | 同上 |
| PDR | [Chapter_01_Introduction.tex:L66](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex#L66) | 正文首次出现未给全称 |
| SNR | [Chapter_04_Fusion.tex:L746](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L746) | 正文首次使用"SNR"缩写，未给全称 |
| LOS/NLOS | [Chapter_04_Fusion.tex:L56](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L56) | 第3章首次出现时给了全称 ✓，但第1章L9"非视距传播"无缩写引入 |
| P95 | [Chapter_04_Fusion.tex:L746](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L746) | 首次使用时无定义说明（后文 §3.6.1 才给出） |

> [!NOTE]
> 按照学位论文规范，每章首次使用缩写应给出全称（摘要中的展开不替代正文）。

---

## 七、中英文混排空格问题

### 7.1 中文正文与英文/数字之间缺少空格

以下为典型案例（非完全列举）：

| 位置 | 问题文本 | 应为 |
|------|----------|------|
| [Ch01:L77](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex#L77) | `利用Radio Map` | `利用 Radio Map` |
| [Ch04:L56](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L56) | `RSSI波动` | `RSSI 波动` |

---

## 八、标点不统一

### 8.1 公式后逗号 / 句号使用

- 部分公式末尾有逗号/句号（如 Chapter_02 大部分公式），部分公式末尾无标点（如 [Ch04:L40](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L40) eq:3_1_fusion_model 末尾无标点）。
- [Ch04:L45](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L45) eq:3_2_weight_constraint 末尾也无标点。
- 建议全文统一：公式在逻辑上属于句子成分时，末尾应加逗号或句号。

### 8.2 破折号使用

- "教师—学生"统一使用中文破折号 `—`（全文一致 ✓）

---

## 九、Section 小标题风格不一致

### 9.1 "本节小结" vs "本章小结"

| 位置 | 标题 | 级别 |
|------|------|------|
| [Ch04:L904](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L904) | `\subsection{本节小结}` | subsection |
| [Ch04:L909](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L909) | `\section{本章小结}` | section |
| [Ch05:L846](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex#L846) | `\subsection{本节小结}` | subsection |
| [Ch05:L851](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex#L851) | `\section{本章小结}` | section |
| [Ch02:L557](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_02_Fundamentals.tex#L557) | `\section{本章小结}` | section（无"本节小结"） |

第3、4章的实验分析节后面各有一个 `\subsection{本节小结}`，而第2章没有"本节小结"，风格不一致。

---

## 十、其他问题

### 10.1 Chapter_02 节编号缺失

[Chapter_02_Fundamentals.tex:L293](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_02_Fundamentals.tex#L293)：

```latex
\end{figure}
\section{智能定位方法基础}
```

`\section{智能定位方法基础}` 前面的 `\end{figure}` 之后缺少一个空行（格式问题，功能不影响但不整洁）。

### 10.2 第4章（Chapter_05_Multi_Target.tex）中§4.4.6"本节小结"文字提到了四个方面

[Chapter_05_Multi_Target.tex:L849](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex#L849)：
> 本节围绕性能对比、掉点鲁棒性、**空间预测可视化**以及**蒸馏机制消融**四个方面，对本文……进行了实验验证。

但实际正文中 subsec:4_4_4（定位结果可视化分析）和 subsec:4_4_5（蒸馏机制消融实验）**均已被注释掉**，并未出现在编译输出中。小结中仍提到"四个方面"需改为"两个方面"。

### 10.3 第3章（Chapter_04_Fusion.tex）§3.4 开头第二句

[Chapter_04_Fusion.tex:L480](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex#L480)：
> 在前述小节中，本文已经分别给出了 PA-DQN 的物理感知状态空间、增量式动作空间以及奖励函数设计。至此……**在前述小节中**，智能体在每一时刻……

连续两句话都以"在前述小节中"开头，属于**重复句首**。

### 10.4 第5章（Chapter_05_Multi_Target.tex）公式缺少编号标签

[Chapter_05_Multi_Target.tex:L349](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex#L349)：
```latex
\end{equation}
```
此处公式（L344–L349）缺少 `\label{}`。

[Chapter_05_Multi_Target.tex:L360](file:///Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex#L360)：同样该公式缺少 `\label{}`。

---

## 汇总清单

| 序号 | 类型 | 严重程度 | 位置 | 简述 |
|------|------|----------|------|------|
| 1 | 重复段落 | 🔴 高 | Ch01:L41–49 | 与 L25–33 完全重复 |
| 2 | 图文不匹配 | 🔴 高 | Ch05:L650 | 图片文件名是"掉点误差对比图"但 caption 写"网络架构示意图" |
| 3 | 注释掉的内容仍被小结引用 | 🟡 中 | Ch05:L849 | 小结说"四个方面"但其中两个已被注释 |
| 4 | 硬编码表号 | 🟡 中 | Ch04:L744 | "表3-3"硬编码且无对应表 |
| 5 | label 编号跳号 | 🟡 中 | Ch01:L71 | `sec:1_4` 跳到 `sec:1_6`，缺 `sec:1_5` |
| 6 | 文件名与章号矛盾 | 🟡 中 | main.tex | 文件名 Chapter_04/05/06 但实际是第3/4/5章 |
| 7 | 符号冲突 | 🟡 中 | Ch05:L216 vs L249 | $\mathcal{B}$ 在同一章表示两个不同概念 |
| 8 | 经验回放池符号不统一 | 🟢 低 | Ch04:L97 vs L643 | $D$ vs $\mathcal{D}$ |
| 9 | DQN target 下标不统一 | 🟢 低 | Ch04:L100 vs Ch02:L383 | $y_i$ vs $y_t$ |
| 10 | 公式末尾标点不统一 | 🟢 低 | 多处 | 部分公式有标点，部分无 |
| 11 | 图片路径写法不统一 | 🟢 低 | Ch02 vs Ch04 | 带/不带 `figures/` 前缀 |
| 12 | 首次缩写未给全称 | 🟢 低 | Ch01:L9 等 | RSSI/AoA/PDR/SNR |
| 13 | 中英文混排空格 | 🟢 低 | Ch01:L77, Ch04:L56 | `RSSI波动` 等 |
| 14 | 重复句首 | 🟢 低 | Ch04:L480 | 连续两句"在前述小节中" |
| 15 | 病句 | 🟢 低 | Ch01:L77 | "利用Radio Map" 拼接不顺 |
| 16 | 公式缺label | 🟢 低 | Ch05:L349,L360 | 两处 equation 无 label |
