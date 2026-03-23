# 大论文低级错误审查报告 v2

> [!IMPORTANT]
> 本报告基于当前源码状态复核，仅核查“低级错误/格式与一致性问题/明显清稿遗留”，不涉及方法内容优劣判断。

---

## 一、复核结论

原报告整体方向基本正确，但混合了三类内容：

1. 当前源码中可以直接确认属实的问题。
2. 方向合理、但更适合作为“风格统一建议”的问题。
3. 已过时、定位不准或不宜继续保留的问题。

本 v2 仅保留经过核实的内容，并对严重程度进行重新整理。

---

## 二、属实且建议优先处理的问题

### 1. 第 1 章 1.2 节存在重复段落

- 文件：
  [Chapter_01_Introduction.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex)
- 复核结果：属实
- 说明：
  第 25–33 行与第 41–49 行大段重复，且第 41 行从“覆盖能力方面存在明显不足”这样的残句起头，明显存在复制粘贴遗留。

### 2. 第 1 章“论文结构安排”中第 4 章描述存在病句

- 文件：
  [Chapter_01_Introduction.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex)
- 位置：
  第 77 行
- 复核结果：属实
- 原文：
  `研究掉点情况下的 利用Radio Map 定位算法与视觉驱动的多目标识别定位算法。`
- 说明：
  存在中英文间缺空格、成分拼接不顺、语句不通顺的问题。

### 3. 第 1 章 label 编号存在跳号

- 文件：
  [Chapter_01_Introduction.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex)
- 复核结果：属实
- 说明：
  `sec:1_4` 之后直接到 `sec:1_6`，缺少 `sec:1_5`。这通常意味着“主要创新点”等小节曾被移除，但 label 风格未同步整理。

### 4. 第 3 章存在硬编码表号

- 文件：
  [Chapter_04_Fusion.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex)
- 位置：
  第 744 行
- 复核结果：属实
- 原文：
  `具体参数如下表3-3所示。`
- 说明：
  此处未使用 `\ref{}` 交叉引用，属于典型硬编码编号问题。

### 5. 第 5 章图文不匹配

- 文件：
  [Chapter_05_Multi_Target.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex)
- 位置：
  第 648–652 行
- 复核结果：属实
- 说明：
  插入的图片文件名为 `8_不同掉点比例下的定位误差对比图.png`，但 caption 写的是“教师—学生网络架构示意图”，文件名与图题明显不一致，应优先核实是否放错图。经过检验没有放错图。

### 6. 第 5 章“本节小结”与正文实际内容不一致

- 文件：
  [Chapter_05_Multi_Target.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex)
- 位置：
  第 849 行
- 复核结果：属实
- 说明：
  该处写“围绕性能对比、掉点鲁棒性、空间预测可视化以及蒸馏机制消融四个方面”，但“空间预测可视化”和“蒸馏机制消融”两部分正文已经被注释掉，没有实际出现在当前编译内容中。

### 7. 第 3 章 DQN 损失符号下标不统一

- 文件：
  [Chapter_04_Fusion.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex)
- 位置：
  第 93–105 行
- 复核结果：属实
- 说明：
  该处使用 `y_i`，而其他位置相关表述中也存在 `y_t`。属于同类变量下标风格不统一。

### 8. 第 3 章经验回放池符号不统一

- 文件：
  [Chapter_04_Fusion.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex)
- 复核结果：属实
- 说明：
  同章中出现 `D`、`\mathcal{D}`、`\mathcal{B}` 等不同记法，分别承担经验回放池或训练数据集含义，符号体系不够统一。

### 9. 第 5 章符号 `\mathcal{B}` 存在语义冲突

- 文件：
  [Chapter_05_Multi_Target.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex)
- 复核结果：属实
- 说明：
  同一章中 `\mathcal{B}` 一处表示建筑物集合，另一处又作为候选目标区域的基符号使用，存在歧义。

### 10. 第 5 章两处公式缺少 `\label{}`

- 文件：
  [Chapter_05_Multi_Target.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_05_Multi_Target.tex)
- 位置：
  第 343–349 行、第 354–360 行
- 复核结果：属实
- 说明：
  两个 `equation` 环境没有对应 `\label{}`，属于明显的交叉引用准备不完整问题。

### 11. 第 3 章存在重复句首

- 文件：
  [Chapter_04_Fusion.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex)
- 位置：
  第 480 行
- 复核结果：属实
- 说明：
  同一句中两次出现“在前述小节中”，属于明显语言重复。

---

## 三、属实但属于风格统一/低优先级问题

### 1. 文件名与实际章号不一致

- 文件：
  [main.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/main.tex)
- 复核结果：属实，但属于维护一致性问题
- 说明：
  当前实际正文是第 3/4/5 章，但文件名仍为 `Chapter_04_Fusion`、`Chapter_05_Multi_Target`、`Chapter_06_Conclusion`。功能上没有问题，但对维护者有误导性。

### 2. 图片路径写法不统一

- 涉及文件：
  [Chapter_02_Fundamentals.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_02_Fundamentals.tex)
  [Chapter_04_Fusion.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex)
- 复核结果：属实，但仅为风格问题
- 说明：
  第 2 章常写 `figures/ch2/...`，第 3 章常写 `ch3/...`。由于 `main.tex` 已设置 `\graphicspath{{figures/}}`，所以功能不受影响，但风格不统一。

### 3. 首次缩写未给全称

- 复核结果：部分属实
- 说明：
  诸如 `RSSI`、`AoA`、`PDR`、`SNR` 等缩写在正文局部位置存在首次出现未展开的问题。属于规范性不足，不是功能错误。

### 4. 中英文混排空格问题

- 复核结果：属实
- 典型例子：
  - [Chapter_01_Introduction.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex) 第 77 行：`利用Radio Map`
  - [Chapter_04_Fusion.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex) 第 56 行：`RSSI波动`
- 说明：
  属于清稿阶段应统一处理的问题。

### 5. 公式末尾标点不统一

- 复核结果：属实
- 说明：
  不同章节对公式末尾逗号/句号的处理风格不统一。建议全文统一，但不构成逻辑错误。

### 6. “本节小结”与“本章小结”风格不一致

- 复核结果：属实，但属于结构风格问题
- 说明：
  第 2 章没有“本节小结”，第 3、4 章实验部分有“本节小结”。属于章节组织风格不完全统一。

---

## 四、不宜继续保留或需收敛表述的内容

### 1. “第 1 章论文结构安排中五章说法有问题”

- 复核结果：不属实
- 说明：
  当前 [Chapter_01_Introduction.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_01_Introduction.tex) 第 72 行“本文共分为五章”与现有 `main.tex` 编译结构一致，应保留，不应继续作为问题项。

### 2. “Chapter_02 节编号缺失”

- 复核结果：不属实
- 说明：
  原报告把 `\end{figure}` 后缺少空行表述为“节编号缺失”，这个定性不准确。实际只是源码排版习惯问题，不影响章节编号或编译输出。

### 3. 原报告中的部分行号已过时

- 复核结果：属实
- 说明：
  你近期已经多次修改 [Chapter_04_Fusion.tex](/Users/miles/Desktop/大论文/BUPTGraduateThesisLatexTemplate/Chapter/Chapter_04_Fusion.tex) 等文件，因此原报告中的个别行号和旧表述已不再精确对应当前源码。后续若继续审查，应以当前源码为准重新定位。

---

## 五、建议的处理顺序

建议按以下顺序处理：

1. 先处理高优先级且会直接影响阅读和引用正确性的问题：
   - Ch01 重复段落
   - Ch05 图文不匹配
   - Ch05 小结与正文不一致
   - Ch04 硬编码表号
   - 公式缺 `label`
2. 再处理符号统一和病句：
   - `y_i / y_t`
   - `D / \mathcal{D} / \mathcal{B}`
   - `\mathcal{B}` 语义冲突
   - “利用Radio Map” 等病句
3. 最后统一低优先级风格项：
   - 空格
   - 公式标点
   - 图片路径写法
   - 小结层级风格

---

## 六、汇总结论

当前这份“格式错误”原报告中，以下问题可以确认属实且值得保留：

- 第 1 章重复段落
- 第 1 章结构安排病句
- 第 1 章 label 跳号
- 第 3 章硬编码表号
- 第 5 章图文不匹配
- 第 5 章小结与实际正文不一致
- 若干符号不统一与公式缺 label

其余不少条目更适合作为统一清稿建议，而不是“错误”。此外，原报告中的个别行号和少数定性已经过时，不建议继续原样使用。
