from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap


ROOT = Path("/Users/miles/Desktop/大论文")
OUT = ROOT / "答辩PPT_图片"
OUT.mkdir(exist_ok=True)

W, H = 1920, 1080
NAVY = "#071a44"
CYAN = "#0d7890"
BODY = "#24334d"
PANEL = "#fbfdff"
BORDER = "#b9cfe7"
AMBER = "#d9902f"
WHITE = "#ffffff"
LIGHT_BG = "#ffffff"
MUTED = "#7b8a98"
LINE_BLUE = "#0d7890"
ICON_BLUE = "#234f83"

FONT_MED = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_LIGHT = "/System/Library/Fonts/STHeiti Light.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT_MED if bold else FONT_LIGHT, size)


def draw_text_wrapped(draw, xy, text, fnt, fill, max_width, line_spacing=8):
    x, y = xy
    lines = []
    current = ""
    for ch in text:
        test = current + ch
        if draw.textbbox((0, 0), test, font=fnt)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = ch
    if current:
        lines.append(current)
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_spacing
    return y


def cover_fit(img, box):
    bx, by, bw, bh = box
    iw, ih = img.size
    scale = max(bw / iw, bh / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - bw) // 2)
    top = max(0, (nh - bh) // 2)
    img = img.crop((left, top, left + bw, top + bh))
    return img


def contain_fit(img, box):
    bx, by, bw, bh = box
    iw, ih = img.size
    scale = min(bw / iw, bh / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def shadow_panel(draw, box, radius=18, fill=PANEL, outline=BORDER, width=2):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1 + 5, y1 + 6, x2 + 5, y2 + 6), radius=radius, fill="#edf2f7")
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def blue_corner_badge(draw, x, y, num, w=92, h=58):
    # Folded-corner number badge used across this slide style.
    draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill="#07356f")
    draw.polygon([(x + w - 30, y), (x + w, y), (x + w, y + h), (x + w - 62, y + h)], fill=PANEL)
    draw.text((x + 16, y + 12), num, font=font(29, True), fill=WHITE)


def sample_card(draw, box, num):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=9, fill=PANEL, outline=BORDER, width=2)
    blue_corner_badge(draw, x1, y1, num)


def draw_question(draw, x, y, text, max_width):
    r = 27
    draw.ellipse((x, y, x + 2 * r, y + 2 * r), outline=LINE_BLUE, width=4)
    draw.text((x + 17, y + 5), "?", font=font(34, True), fill=LINE_BLUE)
    draw.text((x + 76, y), "对应问题：", font=font(24, True), fill=LINE_BLUE)
    draw_text_wrapped(draw, (x + 76, y + 34), text, font(22, True), LINE_BLUE, max_width - 76, line_spacing=4)


def draw_bottom_formula(draw, terms, y=910):
    x1, x2 = 90, 1835
    draw.rounded_rectangle((x1, y, x2, y + 86), radius=10, fill=PANEL, outline=LINE_BLUE, width=2)
    x = x1 + 58
    for i, (kind, label) in enumerate(terms):
        draw_card_icon(draw, x, y + 43, kind)
        draw.text((x + 68, y + 24), label, font=font(24, True), fill=NAVY)
        x += 68 + draw.textbbox((0, 0), label, font=font(24, True))[2] + 46
        if i < len(terms) - 1:
            draw.text((x, y + 21), "+" if i < len(terms) - 2 else "→", font=font(36, True), fill=LINE_BLUE)
            x += 62


def base_slide(title):
    img = Image.new("RGB", (W, H), LIGHT_BG)
    draw = ImageDraw.Draw(img)
    draw.text((60, 52), title, font=font(60, True), fill=NAVY)
    draw.rounded_rectangle((60, 136, 310, 146), radius=4, fill=LINE_BLUE)
    draw.line((60, 146, 1860, 146), fill=LINE_BLUE, width=2)
    return img, draw


def draw_caption(draw, x, y, text):
    draw.text((x, y), text, font=font(18), fill=MUTED)


def draw_card_icon(draw, cx, cy, kind):
    r = 46
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="#6e96c1", width=2)
    if kind == "alert":
        draw.rectangle((cx - 18, cy - 22, cx + 18, cy + 24), outline=ICON_BLUE, width=3)
        draw.line((cx - 8, cy - 8, cx + 8, cy - 8), fill=ICON_BLUE, width=3)
        draw.line((cx - 8, cy + 2, cx + 8, cy + 2), fill=ICON_BLUE, width=3)
        draw.polygon([(cx + 18, cy - 2), (cx + 35, cy + 10), (cx + 18, cy + 22)], outline=AMBER, fill=None)
    elif kind == "uav":
        draw.line((cx - 30, cy, cx + 30, cy), fill=ICON_BLUE, width=3)
        draw.rounded_rectangle((cx - 16, cy - 10, cx + 16, cy + 10), radius=5, outline=ICON_BLUE, width=3)
        for px in [cx - 35, cx + 35]:
            draw.ellipse((px - 8, cy - 8, px + 8, cy + 8), outline=ICON_BLUE, width=3)
        draw.line((cx, cy + 12, cx, cy + 30), fill=ICON_BLUE, width=2)
        draw.arc((cx - 16, cy + 20, cx + 16, cy + 42), 200, 340, fill=CYAN, width=2)
    elif kind == "signal":
        draw.line((cx - 34, cy + 24, cx + 34, cy + 24), fill=ICON_BLUE, width=3)
        for i, h in enumerate([18, 32, 24, 42, 28]):
            x = cx - 28 + i * 14
            draw.line((x, cy + 24, x, cy + 24 - h), fill=ICON_BLUE if i != 3 else AMBER, width=4)
    elif kind == "target":
        draw.ellipse((cx - 24, cy - 24, cx + 24, cy + 24), outline=ICON_BLUE, width=3)
        draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), outline=ICON_BLUE, width=3)
        draw.line((cx - 36, cy, cx + 36, cy), fill=ICON_BLUE, width=3)
        draw.line((cx, cy - 36, cx, cy + 36), fill=ICON_BLUE, width=3)
    elif kind == "challenge":
        draw.line((cx - 30, cy + 18, cx + 30, cy + 18), fill=ICON_BLUE, width=3)
        draw.polygon([(cx, cy - 34), (cx + 30, cy + 18), (cx - 30, cy + 18)], outline=ICON_BLUE, fill=None)
        draw.text((cx - 6, cy - 17), "!", font=font(26, True), fill=AMBER)
    elif kind == "route":
        draw.rectangle((cx - 28, cy - 22, cx + 28, cy + 22), outline=ICON_BLUE, width=3)
        draw.line((cx - 15, cy - 8, cx + 15, cy - 8), fill=ICON_BLUE, width=3)
        draw.line((cx - 15, cy + 6, cx + 8, cy + 6), fill=ICON_BLUE, width=3)


def old_style_info_card(draw, box, icon_kind, heading, body, body_size=23):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=7, fill="#fbfdff", outline="#b9cfe7", width=2)
    cx, cy = x1 + 100, (y1 + y2) // 2
    draw_card_icon(draw, cx, cy, icon_kind)
    draw.line((x1 + 198, y1 + 26, x1 + 198, y2 - 26), fill="#7aa2c7", width=2)
    draw.text((x1 + 246, y1 + 26), heading, font=font(30, True), fill="#071a44")
    draw_text_wrapped(draw, (x1 + 246, y1 + 74), body, font(body_size), "#24334d", x2 - x1 - 285, line_spacing=9)


def slide4():
    img, draw = base_slide("任务边界与无人机平台优势")

    # Three cards, sample style.
    cards = [
        ("01", "任务边界：异常线索后的定位排查", "监管系统或运营商侧通过接入异常、频谱异常或许可数据库比对，获得粗粒度可疑区域；本文进一步研究如何估计可疑终端空间位置。", "alert"),
        ("02", "平台优势：低成本 UAV 机动观测", "无人机具有部署灵活、机动性强和空间视角可调等优势，可进入目标区域上空或周边，对地面终端辐射信号进行空间扫描。", "uav"),
        ("03", "平台约束：载荷、续航与算力受限", "低成本平台难以依赖大规模阵列和高复杂度同步系统，定位方法需要兼顾精度、环境适应性与工程可用性。", "signal"),
    ]
    x0, y0, cw, ch, gap = 60, 178, 850, 204, 26
    for i, (num, head, body, kind) in enumerate(cards):
        y = y0 + i * (ch + gap)
        sample_card(draw, (x0, y, x0 + cw, y + ch), num)
        draw_card_icon(draw, x0 + 105, y + 102, kind)
        draw.line((x0 + 196, y + 38, x0 + 196, y + ch - 38), fill="#7aa2c7", width=2)
        draw.text((x0 + 240, y + 33), head, font=font(30, True), fill=NAVY)
        draw_text_wrapped(draw, (x0 + 240, y + 86), body, font(22), BODY, cw - 280, line_spacing=8)

    # Right cited figure
    fig_path = ROOT / "BUPTGraduateThesisLatexTemplate/figures/ch2/1_无人机机动观测下非法终端定位场景示意图.jpg"
    fig = Image.open(fig_path).convert("RGB")
    rx, ry, rw, rh = 965, 178, 895, 664
    draw.rounded_rectangle((rx, ry, rx + rw, ry + rh), radius=9, fill=PANEL, outline=BORDER, width=2)
    draw.line((rx + 34, ry + 80, rx + rw - 34, ry + 80), fill="#d8e6f2", width=1)
    draw.text((rx + 34, ry + 28), "无人机机动观测定位场景", font=font(30, True), fill=NAVY)
    # No visible citation caption; this is the actual thesis figure embedded.
    fit = contain_fit(fig, (rx + 36, ry + 104, rw - 72, rh - 132))
    fx = rx + (rw - fit.size[0]) // 2
    fy = ry + 104 + (rh - 132 - fit.size[1]) // 2
    img.paste(fit, (fx, fy))

    draw_bottom_formula(
        draw,
        [
            ("alert", "异常线索"),
            ("uav", "UAV 机动观测"),
            ("signal", "多源观测提取"),
            ("target", "非合作定位排查"),
        ],
        y=902,
    )

    img.save(OUT / "slide_04.png", quality=95)


def slide5():
    img, draw = base_slide("复杂城市环境下的关键挑战")

    cards = [
        ("01", "非合作信号稳定性不足", "终端不提供可信位置、同步辅助或协议配合；卫星互联网信号还可能受到时钟漂移、调度跳变与传播扰动影响。", "如何构建稳健观测与推断方法？"),
        ("02", "单目标观测质量动态波动", "RSSI、AOA、TDOA 等观测量在遮挡、多径和非视距条件下误差特性随时间变化，固定权重融合难以适应。", "如何引入传播先验实现自适应调权？"),
        ("03", "多目标协同定位鲁棒性不足", "目标数量不确定、UAV 稀疏扫描、局部多峰、多径鬼影与节点掉点并存，导致区域判别与协同定位困难。", "如何在残缺观测下保持稳定定位能力？"),
    ]
    x0, y0, cw, ch, gap = 58, 180, 565, 690, 32
    for idx, (num, title, body, q) in enumerate(cards):
        x = x0 + idx * (cw + gap)
        sample_card(draw, (x, y0, x + cw, y0 + ch), num)

        # Top illustration area.
        iy = y0 + 44
        if idx == 0:
            # clock + waveform + warning
            cx, cy = x + 240, iy + 70
            draw.arc((cx - 58, cy - 58, cx + 58, cy + 58), 25, 330, fill=ICON_BLUE, width=6)
            draw.line((cx, cy, cx, cy - 45), fill=ICON_BLUE, width=6)
            draw.line((cx, cy, cx + 35, cy + 25), fill=ICON_BLUE, width=6)
            wx = x + 335
            pts = [(wx, cy), (wx+22, cy), (wx+34, cy-35), (wx+48, cy+22), (wx+62, cy-18), (wx+78, cy+12), (wx+92, cy-8), (wx+112, cy)]
            draw.line(pts, fill=ICON_BLUE, width=4)
            draw.polygon([(x+315, iy+150), (x+355, iy+84), (x+395, iy+150)], outline=AMBER, fill=None)
            draw.text((x+344, iy+111), "!", font=font(30, True), fill=AMBER)
        elif idx == 1:
            # observation sources to bar chart
            sx = x + 118
            for k, kind in enumerate(["signal", "uav", "alert"]):
                draw_card_icon(draw, sx, iy + 52 + k * 42, kind)
                draw.line((sx + 50, iy + 52 + k * 42, x + 250, iy + 76), fill="#6e96c1", width=2)
            draw.rounded_rectangle((x + 250, iy + 22, x + 420, iy + 150), radius=7, outline=ICON_BLUE, width=3)
            for k, h in enumerate([90, 72, 58, 45, 34, 26]):
                bx = x + 278 + k * 22
                draw.rectangle((bx, iy + 135 - h, bx + 14, iy + 135), fill=[ICON_BLUE, LINE_BLUE, "#70a8c4", "#9dbdd1", "#bdd0dc", "#d2dee6"][k])
            draw.arc((x + 315, iy + 56, x + 425, iy + 150), 205, 330, fill=ICON_BLUE, width=2)
            draw.text((x + 442, iy + 60), "权重\n随时间\n变化", font=font(21, True), fill="#071a44")
        else:
            # UAV network and grid
            grid_y = iy + 120
            for gx in range(6):
                draw.line((x + 135 + gx*48, grid_y, x + 135 + gx*48, grid_y + 70), fill="#b8c8d8", width=1)
            for gy in range(4):
                draw.line((x + 135, grid_y + gy*23, x + 375, grid_y + gy*23), fill="#b8c8d8", width=1)
            for px, py, col in [(x+175, grid_y+35, ICON_BLUE), (x+245, grid_y+18, ICON_BLUE), (x+318, grid_y+42, AMBER), (x+365, grid_y+20, ICON_BLUE)]:
                draw.ellipse((px-8, py-8, px+8, py+8), fill=col)
                draw.line((px, py+8, px, py+24), fill=col, width=3)
            for ux, uy in [(x+150, iy+48), (x+250, iy+35), (x+365, iy+55), (x+452, iy+40)]:
                draw.line((ux-18, uy, ux+18, uy), fill=ICON_BLUE, width=3)
                draw.rounded_rectangle((ux-9, uy-6, ux+9, uy+6), radius=3, outline=ICON_BLUE, width=2)
                draw.line((ux, uy+8, ux, uy+38), fill="#6e96c1", width=1)
            draw.line((x+160, iy+48, x+365, iy+55), fill="#6e96c1", width=2)
            draw.line((x+250, iy+35, x+452, iy+40), fill="#6e96c1", width=2)

        draw.text((x + 36, y0 + 230), "挑战" + ["一", "二", "三"][idx] + "：" + title, font=font(28, True), fill="#071a44")
        draw_text_wrapped(draw, (x + 36, y0 + 292), body, font(22), "#24334d", cw - 72, line_spacing=11)
        draw.line((x + 32, y0 + 422, x + cw - 32, y0 + 422), fill=LINE_BLUE, width=1)
        # Dotted line overlay effect.
        for dx in range(x + 32, x + cw - 32, 16):
            draw.line((dx, y0 + 422, dx + 8, y0 + 422), fill=WHITE, width=2)

        draw_question(draw, x + 36, y0 + 450, q, cw - 72)

        # Bottom mini illustration, matching sample visual density.
        dy = y0 + 585
        if idx == 0:
            draw_card_icon(draw, x + 80, dy + 34, "signal")
            draw.line((x + 145, dy + 52, x + 365, dy + 52), fill=ICON_BLUE, width=2)
            # jagged line
            pts = []
            for k in range(7):
                pts.append((x + 160 + k*32, dy + 52 + ([0, -28, 10, -16, 24, -30, 6][k])))
            draw.line(pts, fill=ICON_BLUE, width=3)
            draw.line((x + 365, dy + 52, x + 450, dy + 52), fill=ICON_BLUE, width=2)
            draw_card_icon(draw, x + 480, dy + 34, "target")
        elif idx == 1:
            # city -> variation -> bars
            for bx, h in [(x+50, 40), (x+75, 70), (x+105, 52)]:
                draw.rectangle((bx, dy+70-h, bx+18, dy+70), outline=ICON_BLUE, width=2)
            draw.line((x+140, dy+45, x+210, dy+18), fill=LINE_BLUE, width=2)
            draw.line((x+210, dy+18, x+260, dy+64), fill=LINE_BLUE, width=2)
            for k, h in enumerate([72, 55, 44, 34, 24]):
                draw.rectangle((x+365+k*24, dy+72-h, x+382+k*24, dy+72), fill=[ICON_BLUE, LINE_BLUE, "#70a8c4", "#9dbdd1", "#bdd0dc"][k])
            draw.line((x+350, dy+72, x+510, dy+72), fill=ICON_BLUE, width=2)
        else:
            # grid + gaussian peaks
            for gx in range(8):
                draw.line((x + 40 + gx*24, dy+4, x + 40 + gx*24, dy+90), fill="#b8c8d8", width=1)
            for gy in range(5):
                draw.line((x + 40, dy+4+gy*22, x+208, dy+4+gy*22), fill="#b8c8d8", width=1)
            for px, py, col in [(x+70, dy+38, ICON_BLUE), (x+120, dy+20, ICON_BLUE), (x+160, dy+60, AMBER)]:
                draw.ellipse((px-8, py-8, px+8, py+8), fill=col)
            # peaks
            base = dy+82
            for k, cxp in enumerate([x+290, x+390, x+485]):
                pts = [(cxp-35, base), (cxp-18, base-20), (cxp, base-75 if k==1 else base-55), (cxp+18, base-20), (cxp+35, base)]
                draw.line(pts, fill=ICON_BLUE if k != 1 else AMBER, width=3)

    draw_bottom_formula(
        draw,
        [
            ("challenge", "复杂传播环境"),
            ("signal", "非合作观测"),
            ("uav", "低成本平台约束"),
            ("target", "定位精度、尾部误差与工程鲁棒性面临挑战"),
        ],
        y=910,
    )

    img.save(OUT / "slide_05.png", quality=95)


def slide6():
    img, draw = base_slide("论文研究内容与技术路线")

    # Left content modules
    modules = [
        ("01", "系统建模与理论基础", ["非合作终端排查任务建模", "复杂城市传播机理分析", "RSSI/AOA/TDOA/PDR 与智能方法基础"]),
        ("02", "单目标自适应融合定位", ["构建 RSSI、AOA、PDR 多源融合模型", "提取 RMS 时延扩展、RSSI 方差、几何一致性", "设计 PA-DQN 动态调权策略"]),
        ("03", "多目标协同鲁棒定位", ["离散 UAV 扫描观测映射为 Radio Map", "利用区域结构缓解局部多峰与多径鬼影", "教师—学生知识蒸馏应对节点掉点"]),
    ]
    lx, ly, lw, mh, gap = 60, 178, 760, 190, 28
    for i, (num, head, bullets) in enumerate(modules):
        y = ly + i * (mh + gap)
        sample_card(draw, (lx, y, lx + lw, y + mh), num)
        draw_card_icon(draw, lx + 112, y + 100, "route" if i == 0 else ("signal" if i == 1 else "target"))
        draw.line((lx + 196, y + 34, lx + 196, y + mh - 34), fill="#7aa2c7", width=2)
        draw.text((lx + 232, y + 28), head, font=font(28, True), fill=NAVY)
        by = y + 72
        for b in bullets:
            draw.ellipse((lx + 236, by + 8, lx + 248, by + 20), fill=LINE_BLUE)
            draw_text_wrapped(draw, (lx + 266, by), b, font(20), BODY, lw - 290, line_spacing=4)
            by += 35

    # Right cited thesis structure figure
    fig_path = ROOT / "BUPTGraduateThesisLatexTemplate/figures/ch1/全文结构图.png"
    fig = Image.open(fig_path).convert("RGB")
    rx, ry, rw, rh = 875, 178, 985, 640
    draw.rounded_rectangle((rx, ry, rx + rw, ry + rh), radius=9, fill=PANEL, outline=BORDER, width=2)
    draw.text((rx + 34, ry + 28), "全文研究内容组织关系", font=font(30, True), fill=NAVY)
    draw.line((rx + 34, ry + 80, rx + rw - 34, ry + 80), fill="#d8e6f2", width=1)
    # No visible citation caption; this is the actual thesis figure embedded.
    fit = contain_fit(fig, (rx + 34, ry + 100, rw - 68, rh - 126))
    fx = rx + (rw - fit.size[0]) // 2
    fy = ry + 100 + (rh - 126 - fit.size[1]) // 2
    img.paste(fit, (fx, fy))

    # Bottom goal box
    draw.rounded_rectangle((60, 900, 1860, 986), radius=10, fill=PANEL, outline=LINE_BLUE, width=2)
    draw_card_icon(draw, 125, 943, "route")
    goal = "总体目标：提升低成本无人机平台在复杂城市环境中的定位精度、环境适应性与工程鲁棒性。"
    draw_text_wrapped(draw, (200, 918), goal, font(28, True), NAVY, 1580, line_spacing=6)

    img.save(OUT / "slide_06.png", quality=95)


if __name__ == "__main__":
    slide4()
    slide5()
    slide6()
    print(OUT / "slide_04.png")
    print(OUT / "slide_05.png")
    print(OUT / "slide_06.png")
