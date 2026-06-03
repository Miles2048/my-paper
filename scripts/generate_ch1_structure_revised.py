from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "BUPTGraduateThesisLatexTemplate" / "figures" / "ch1" / "全文结构图_修订版.png"


W, H = 2400, 1350
BG = "white"
LINE = (45, 45, 45)
TEXT = (20, 20, 20)
SUB = (35, 35, 35)
BLUE = (229, 239, 252)
BLUE_LINE = (95, 135, 190)
GREEN = (232, 244, 232)
GREEN_LINE = (93, 151, 93)
ORANGE = (252, 239, 225)
ORANGE_LINE = (205, 142, 82)
GRAY = (245, 245, 245)
GRAY_LINE = (110, 110, 110)

FONT_REG = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


F_TITLE = font(34, True)
F_BOX = font(30, True)
F_SMALL = font(25)
F_SMALL_B = font(25, True)
F_TINY = font(23)


img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)


def round_box(xy, fill, outline, r=16, width=3):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def center_text(text, box, fnt, fill=TEXT, dy=0):
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((x1 + x2 - tw) / 2, (y1 + y2 - th) / 2 + dy), text, font=fnt, fill=fill)


def draw_wrapped(text, box, fnt, fill=TEXT, max_chars=24, line_gap=8, bullet=False):
    x1, y1, x2, y2 = box
    lines = []
    for para in text.split("\n"):
        chunks = textwrap.wrap(para, width=max_chars, break_long_words=False, replace_whitespace=False)
        lines.extend(chunks if chunks else [""])
    line_h = fnt.getbbox("国")[3] - fnt.getbbox("国")[1] + line_gap
    y = y1
    for i, line in enumerate(lines):
        prefix = "• " if bullet and i == 0 else "  " if bullet else ""
        draw.text((x1, y), prefix + line, font=fnt, fill=fill)
        y += line_h


def arrow(x1, y1, x2, y2):
    draw.line((x1, y1, x2, y2), fill=LINE, width=3)
    # simple triangular arrow head
    if abs(y2 - y1) >= abs(x2 - x1):
        s = 13
        if y2 >= y1:
            pts = [(x2, y2), (x2 - s, y2 - s), (x2 + s, y2 - s)]
        else:
            pts = [(x2, y2), (x2 - s, y2 + s), (x2 + s, y2 + s)]
    else:
        s = 13
        if x2 >= x1:
            pts = [(x2, y2), (x2 - s, y2 - s), (x2 - s, y2 + s)]
        else:
            pts = [(x2, y2), (x2 + s, y2 - s), (x2 + s, y2 + s)]
    draw.polygon(pts, fill=LINE)


margin = 85
top = (margin, 45, W - margin, 165)
problem1 = (margin, 205, W // 2 - 35, 335)
problem2 = (W // 2 + 35, 205, W - margin, 335)
ch2 = (margin, 385, W - margin, 555)
ch3 = (margin, 620, W // 2 - 35, 940)
ch4 = (W // 2 + 35, 620, W - margin, 940)
ch5 = (430, 1045, W - 430, 1190)

round_box(top, GRAY, GRAY_LINE)
center_text("第1章  绪论", (top[0], top[1] + 10, top[2], top[1] + 58), F_BOX)
draw_wrapped(
    "研究背景与意义        国内外研究现状        现有研究问题        主要研究内容与结构安排",
    (top[0] + 135, top[1] + 72, top[2] - 80, top[3] - 10),
    F_SMALL,
    max_chars=200,
)

round_box(problem1, BLUE, BLUE_LINE)
center_text("核心问题一：单目标自适应融合定位", (problem1[0], problem1[1] + 8, problem1[2], problem1[1] + 48), F_SMALL_B)
draw_wrapped(
    "非合作信号稳定性不足，复杂传播环境下 RSSI / AOA / PDR 观测质量动态波动，如何实现稳健观测与自适应融合？",
    (problem1[0] + 34, problem1[1] + 58, problem1[2] - 30, problem1[3] - 15),
    F_TINY,
    max_chars=36,
)

round_box(problem2, ORANGE, ORANGE_LINE)
center_text("核心问题二：多目标协同鲁棒定位", (problem2[0], problem2[1] + 8, problem2[2], problem2[1] + 48), F_SMALL_B)
draw_wrapped(
    "目标数量不确定、稀疏扫描、多径伪峰和无人机节点失效并存，如何实现多目标区域级判别与鲁棒定位？",
    (problem2[0] + 34, problem2[1] + 58, problem2[2] - 30, problem2[3] - 15),
    F_TINY,
    max_chars=36,
)

round_box(ch2, GRAY, GRAY_LINE)
center_text("第2章  系统模型与定位理论基础", (ch2[0], ch2[1] + 10, ch2[2], ch2[1] + 58), F_BOX)
draw_wrapped(
    "无人机载平台定位系统建模        复杂城市传播机理分析\n多源定位观测模型：RSSI / AOA / TOA / TDOA / PDR        强化学习与知识蒸馏基础",
    (ch2[0] + 150, ch2[1] + 75, ch2[2] - 120, ch2[3] - 20),
    F_SMALL,
    max_chars=95,
)

round_box(ch3, BLUE, BLUE_LINE)
center_text("第3章  基于物理感知深度强化学习的单目标自适应融合定位方法", (ch3[0], ch3[1] + 12, ch3[2], ch3[1] + 58), F_SMALL_B)
draw_wrapped(
    "多源定位分支：RSSI、AOA、PDR 分别生成候选位置\n物理感知状态：RMS 时延扩展、RSSI 方差、RF-PDR 一致性、历史权重\nPA-DQN 决策：输出增量式调权动作，动态更新融合权重\n目标：提升复杂环境下单目标连续定位精度与轨迹稳定性",
    (ch3[0] + 48, ch3[1] + 82, ch3[2] - 42, ch3[3] - 38),
    F_SMALL,
    max_chars=37,
    bullet=True,
)

round_box(ch4, ORANGE, ORANGE_LINE)
center_text("第4章  基于空间信号表征的多目标协同鲁棒定位方法", (ch4[0], ch4[1] + 12, ch4[2], ch4[1] + 58), F_SMALL_B)
draw_wrapped(
    "多目标并发定位场景建模：信号叠加、局部多峰与多径伪峰\nRadio Map 信号成像：由离散 RSSI 观测转为区域级空间判别\n教师-学生 GAT 知识蒸馏：适应无人机节点失效与残缺观测\n目标：提升多目标协同定位鲁棒性",
    (ch4[0] + 48, ch4[1] + 82, ch4[2] - 42, ch4[3] - 38),
    F_SMALL,
    max_chars=37,
    bullet=True,
)

round_box(ch5, GREEN, GREEN_LINE)
center_text("第5章  结论与展望", (ch5[0], ch5[1] + 14, ch5[2], ch5[1] + 60), F_BOX)
draw_wrapped(
    "全文工作总结        主要研究结论        未来研究展望",
    (ch5[0] + 210, ch5[1] + 82, ch5[2] - 160, ch5[3] - 20),
    F_SMALL,
    max_chars=100,
)

# vertical arrows
arrow((top[0] + top[2]) // 2, top[3], (top[0] + top[2]) // 2, problem1[1] - 18)
draw.line((W // 2, problem1[1] - 18, W // 2, problem1[1] - 18), fill=LINE, width=3)
arrow(W // 2, problem1[1] - 18, (problem1[0] + problem1[2]) // 2, problem1[1] - 2)
arrow(W // 2, problem2[1] - 18, (problem2[0] + problem2[2]) // 2, problem2[1] - 2)

arrow((problem1[0] + problem1[2]) // 2, problem1[3], (problem1[0] + problem1[2]) // 2, ch2[1] - 3)
arrow((problem2[0] + problem2[2]) // 2, problem2[3], (problem2[0] + problem2[2]) // 2, ch2[1] - 3)
arrow((ch2[0] + ch2[2]) // 2, ch2[3], (ch2[0] + ch2[2]) // 2, ch3[1] - 18)
arrow(W // 2, ch3[1] - 18, (ch3[0] + ch3[2]) // 2, ch3[1] - 3)
arrow(W // 2, ch4[1] - 18, (ch4[0] + ch4[2]) // 2, ch4[1] - 3)
arrow((ch3[0] + ch3[2]) // 2, ch3[3], (ch3[0] + ch3[2]) // 2, ch5[1] - 25)
arrow((ch4[0] + ch4[2]) // 2, ch4[3], (ch4[0] + ch4[2]) // 2, ch5[1] - 25)
draw.line(((ch3[0] + ch3[2]) // 2, ch5[1] - 25, (ch4[0] + ch4[2]) // 2, ch5[1] - 25), fill=LINE, width=3)
arrow(W // 2, ch5[1] - 25, W // 2, ch5[1] - 3)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, quality=95)
print(OUT)
