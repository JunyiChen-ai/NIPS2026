"""Generate PPTX with proper spacing — no crowding."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

PRIMARY = RGBColor(0x8B, 0x5C, 0xF6)
ACCENT = RGBColor(0x25, 0x63, 0xEB)
GREEN = RGBColor(0x05, 0x96, 0x69)
RED = RGBColor(0xDC, 0x26, 0x26)
DARK = RGBColor(0x1E, 0x1E, 0x1E)
GRAY = RGBColor(0x6B, 0x72, 0x80)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHTBG = RGBColor(0xF3, 0xF4, 0xF6)
ORANGE = RGBColor(0xEA, 0x58, 0x0C)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

def add_slide():
    return prs.slides.add_slide(prs.slide_layouts[6])

def title_box(slide, text):
    tx = slide.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11.5), Inches(0.7))
    p = tx.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = PRIMARY

def add_text(slide, left, top, width, height, text, size=20, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    tx = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align
    return tf

def bullets(slide, left, top, width, items, size=20, spacing=12):
    """items = list of (text, {opts}) or str. Good spacing by default."""
    tx = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.5))
    tf = tx.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        txt, opts = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = txt
        p.font.size = Pt(opts.get("s", size))
        p.font.bold = opts.get("b", False)
        p.font.color.rgb = opts.get("c", DARK)
        p.alignment = opts.get("a", PP_ALIGN.LEFT)
        p.space_before = Pt(opts.get("sb", spacing))
        p.space_after = Pt(4)

def table(slide, left, top, width, row_h, headers, rows, hcolor=PRIMARY):
    nr, nc = len(rows)+1, len(headers)
    t = slide.shapes.add_table(nr, nc, Inches(left), Inches(top), Inches(width), Inches(row_h * nr)).table
    for j, h in enumerate(headers):
        c = t.cell(0, j); c.text = h
        for p in c.text_frame.paragraphs:
            p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = WHITE; p.alignment = PP_ALIGN.CENTER
        c.fill.solid(); c.fill.fore_color.rgb = hcolor
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i+1, j); c.text = str(val)
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(12); p.font.color.rgb = DARK; p.alignment = PP_ALIGN.CENTER
            if i % 2 == 0: c.fill.solid(); c.fill.fore_color.rgb = LIGHTBG

# ================================================================
# 1. Title
# ================================================================
s = add_slide()
add_text(s, 1, 2.0, 11, 0.8, "Multi-View Internal State Fusion", 42, True, PRIMARY, PP_ALIGN.CENTER)
add_text(s, 1, 2.9, 11, 0.8, "for LLM Probing", 42, True, PRIMARY, PP_ALIGN.CENTER)
add_text(s, 1, 4.3, 11, 0.5, "NeurIPS 2026 — Progress Report", 24, False, ACCENT, PP_ALIGN.CENTER)
add_text(s, 1, 5.2, 11, 0.5, "Junyi Chen  ·  April 2026", 20, False, GRAY, PP_ALIGN.CENTER)

# ================================================================
# 2. Problem
# ================================================================
s = add_slide()
title_box(s, "Problem: Probing Methods Are Fragmented")

bullets(s, 0.8, 1.3, 5, [
    ("What is probing?", {"s": 24, "b": True, "c": ACCENT, "sb": 0}),
    "给LLM一个prompt，提取内部状态，",
    "训练分类器预测：",
    ("  • 回答是否正确？", {"sb": 16}),
    "  • 是否需要工具？",
    "  • 是否在幻觉？",
    "  • 问题有多难？",
], size=20, spacing=8)

bullets(s, 7, 1.3, 5.5, [
    ("现状", {"s": 24, "b": True, "c": RED, "sb": 0}),
    "12+ probing方法各自为政：",
    ("  • 每个用不同的内部信号", {"sb": 16}),
    "  • 每个在不同数据集上评测",
    "  • 从不跟其他方法融合",
    ("", {"sb": 20}),
    ("我们的问题:", {"s": 22, "b": True, "c": PRIMARY}),
    ("能否融合多种probing方法,", {"s": 22, "b": True, "c": PRIMARY}),
    ("超过任何单一方法？", {"s": 22, "b": True, "c": PRIMARY}),
], size=20, spacing=8)

# ================================================================
# 3. Type A: Raw features from LLM
# ================================================================
s = add_slide()
title_box(s, "特征来源 (1/2): 从LLM直接提取的Raw内部状态")
add_text(s, 0.8, 1.1, 11, 0.4, "对每个样本，跑Qwen2.5-7B forward pass，在每一层放hook截取：", 18, False, GRAY)

bullets(s, 0.8, 1.7, 3.5, [
    ("表示类", {"s": 22, "b": True, "c": PRIMARY, "sb": 0}),
    ("Hidden state vectors", {"s": 16, "c": GRAY, "sb": 2}),
    ("", {"sb": 8}),
    ("input_last_token_hidden", {"s": 17, "b": True}),
    ("30层 × 3584维", {"s": 15, "c": GRAY, "sb": 2}),
    ("最后一个prompt token的表示", {"s": 15, "c": GRAY, "sb": 2}),
    ("", {"sb": 8}),
    ("input_mean_pool_hidden", {"s": 17, "b": True}),
    ("30层 × 3584维", {"s": 15, "c": GRAY, "sb": 2}),
    ("所有prompt token的平均", {"s": 15, "c": GRAY, "sb": 2}),
    ("", {"sb": 8}),
    ("gen侧同理 × 2", {"s": 17}),
], size=18, spacing=4)

bullets(s, 4.8, 1.7, 3.8, [
    ("注意力类", {"s": 22, "b": True, "c": ACCENT, "sb": 0}),
    ("Attention patterns", {"s": 16, "c": GRAY, "sb": 2}),
    ("", {"sb": 8}),
    ("per_head_activation", {"s": 17, "b": True}),
    ("28层 × 28头 × 128维", {"s": 15, "c": GRAY, "sb": 2}),
    ("", {"sb": 8}),
    ("attn_stats (skew/entropy/diag)", {"s": 17, "b": True}),
    ("28层 × 28头 × 3", {"s": 15, "c": GRAY, "sb": 2}),
    ("", {"sb": 8}),
    ("attn_value_norms", {"s": 17, "b": True}),
    ("28层 × 28头 × 变长", {"s": 15, "c": GRAY, "sb": 2}),
    ("", {"sb": 8}),
    ("gen侧attn_stats × 1", {"s": 17}),
], size=18, spacing=4)

bullets(s, 9.2, 1.7, 3.5, [
    ("置信度类", {"s": 22, "b": True, "c": GREEN, "sb": 0}),
    ("Logit distribution stats", {"s": 16, "c": GRAY, "sb": 2}),
    ("", {"sb": 8}),
    ("input_logit_stats", {"s": 17, "b": True}),
    ("entropy", {"s": 15, "c": GRAY, "sb": 2}),
    ("max_prob", {"s": 15, "c": GRAY, "sb": 2}),
    ("logsumexp", {"s": 15, "c": GRAY, "sb": 2}),
    ("", {"sb": 8}),
    ("gen_logit_stats", {"s": 17, "b": True}),
    ("同上 × 3个标量", {"s": 15, "c": GRAY, "sb": 2}),
], size=18, spacing=4)

add_text(s, 0.8, 6.5, 11.5, 0.4, "共 10 种 raw feature source，覆盖模型的表示、注意力、置信度三个计算维度", 18, True, PRIMARY, PP_ALIGN.CENTER)

# ================================================================
# 4. Type B: Baseline method outputs
# ================================================================
s = add_slide()
title_box(s, "特征来源 (2/2): 12个Baseline方法的处理输出")
add_text(s, 0.8, 1.1, 11, 0.4, "每个方法拿上述raw特征，做自己定义的后处理，输出预测概率：", 18, False, GRAY)

table(s, 0.8, 1.7, 11.5, 0.38,
    ["方法", "来源 (用了哪个raw特征)", "后处理方式", "选层策略"],
    [
        ["LR Probe", "input_last_token_hidden", "直接训LR", "选val上最好的1层"],
        ["PCA+LR", "input_last_token_hidden", "PCA降维 → LR", "选最好的1层"],
        ["ITI", "per_head_activation", "找informative head → probe", "选最好的1层1头"],
        ["KB MLP", "input_last_token_hidden", "2层MLP", "固定中间层"],
        ["Attn Satisfies", "attn_value_norms", "直接作为scorer", "搜索层+头组合"],
        ["LLM-Check", "attn_stats (diagonal)", "注意力对角线均值", "搜索最好层"],
        ["SEP", "gen_last_token_hidden", "选层范围 → LR", "搜索最好层范围"],
        ["STEP", "gen_per_token_hidden", "MLP scorer", "最后1层"],
        ["MM Probe", "input_last_token_hidden", "均值中心化 → 方向投影", "选最好层"],
        ["CoE/SeaKR/LID", "各种geometric scores", "无监督打分", "无需训练"],
    ])

bullets(s, 0.8, 6.0, 11.5, [
    ("共同点: 每个方法只用了部分raw特征，只选了少数几层，错误模式各不相同", {"s": 19, "b": True, "c": RED, "a": PP_ALIGN.CENTER, "sb": 0}),
    ("我们: 融合其中7个方法的OOF输出作为额外的feature source", {"s": 18, "c": PRIMARY, "a": PP_ALIGN.CENTER, "sb": 8}),
], spacing=4)

# ================================================================
# 5. Method: Stage 1
# ================================================================
s = add_slide()
title_box(s, "方法 Stage 1: 对每种Raw特征的每一层训练Probe")

bullets(s, 0.8, 1.3, 11, [
    ("以 input_last_token_hidden (30层 × 3584维) 为例：", {"s": 20, "c": GRAY, "sb": 0}),
], spacing=4)

bullets(s, 0.8, 2.0, 11, [
    ("Layer 0:  features (3584d) → Scaler → PCA (512d) → C调参 → 5-fold LR → OOF概率", {"s": 18, "sb": 0}),
    ("Layer 2:  features (3584d) → Scaler → PCA (512d) → C调参 → 5-fold LR → OOF概率", {"s": 18}),
    ("Layer 4:  ...", {"s": 18}),
    ("Layer 28: features (3584d) → Scaler → PCA (512d) → C调参 → 5-fold LR → OOF概率", {"s": 18}),
], spacing=10)

bullets(s, 0.8, 4.0, 11, [
    ("→ 这一个source产出 15层 × K类 个OOF概率值", {"s": 20, "b": True, "c": PRIMARY, "sb": 0}),
    ("", {"sb": 16}),
    ("对全部10种raw features重复同样操作 (层间距根据feature type调整)", {"s": 19}),
    ("加上7个baseline方法的OOF输出", {"s": 19}),
    ("", {"sb": 16}),
    ("OOF = Out-of-Fold: 每个样本的预测来自没见过它的模型，保证无leakage", {"s": 17, "c": ACCENT}),
], spacing=8)

# ================================================================
# 6. Method: Stage 2
# ================================================================
s = add_slide()
title_box(s, "方法 Stage 2: 拼接所有OOF概率 → Meta分类器")

bullets(s, 0.8, 1.3, 11, [
    ("把所有view的OOF概率横向拼接:", {"s": 20, "b": True, "sb": 0}),
], spacing=4)

bullets(s, 1.2, 2.0, 10, [
    ("repr_input_last:   15层 × K类", {"s": 17, "c": PRIMARY, "sb": 0}),
    ("repr_input_mean:   15层 × K类", {"s": 17, "c": PRIMARY}),
    ("repr_gen_last:     15层 × K类", {"s": 17, "c": PRIMARY}),
    ("repr_gen_mean:     15层 × K类", {"s": 17, "c": PRIMARY}),
    ("attn_head_act:      7层 × K类", {"s": 17, "c": ACCENT}),
    ("attn_stats + vnorms + gen_stats: 84层 × K类", {"s": 17, "c": ACCENT}),
    ("conf_input + conf_gen: 2 × K类", {"s": 17, "c": GREEN}),
    ("7个baseline方法: 7 × K类", {"s": 17, "c": ORANGE}),
], spacing=6)

bullets(s, 0.8, 5.0, 11, [
    ("→ 共 300~800 个 meta-features (取决于K=类别数)", {"s": 20, "b": True, "sb": 0}),
    ("", {"sb": 12}),
    ("→ StandardScaler → Ridge Logistic Regression (C用CV选) → 最终预测", {"s": 22, "b": True, "c": PRIMARY}),
    ("", {"sb": 12}),
    ("全部线性模型。零per-dataset人工调参。", {"s": 20, "c": GREEN, "b": True}),
], spacing=8)

# ================================================================
# 7. Experiment Setup
# ================================================================
s = add_slide()
title_box(s, "Experiment Setup")

bullets(s, 0.8, 1.3, 5, [
    ("模型", {"s": 22, "b": True, "c": ACCENT, "sb": 0}),
    ("Qwen2.5-7B-Instruct", {"sb": 8}),
    ("bfloat16, 2×A100", {"s": 17, "c": GRAY}),
    ("2-pass extraction (generate → replay)", {"s": 17, "c": GRAY}),
    ("", {"sb": 20}),
    ("Baselines: 12个复现方法", {"s": 22, "b": True, "c": ACCENT}),
    ("全部忠实复现自原论文代码", {"sb": 8}),
    ("经Codex代码审核", {"s": 17, "c": GRAY}),
], size=19, spacing=6)

bullets(s, 7, 1.3, 5.5, [
    ("9个分类数据集", {"s": 22, "b": True, "c": ACCENT, "sb": 0}),
    ("", {"sb": 8}),
    ("Binary (5个):", {"sb": 8}),
    ("  GoT Cities, MetaTool, RetrievalQA,", {"s": 17}),
    ("  FAVA, RAGTruth", {"s": 17}),
    ("", {"sb": 8}),
    ("Multi-class (4个):", {}),
    ("  common_claim (3c), When2Call (3c),", {"s": 17}),
    ("  E2H AMC (3c), E2H AMC (5c)", {"s": 17}),
    ("", {"sb": 12}),
    ("每个数据集 800–3,500 训练样本", {}),
    ("指标: AUROC + bootstrap 95% CI", {}),
], size=19, spacing=6)

# ================================================================
# 8. Results
# ================================================================
s = add_slide()
title_box(s, "Results")

table(s, 0.8, 1.2, 11.5, 0.42,
    ["Dataset", "Task", "Best Single Probe", "Our Fusion", "Delta"],
    [
        ["GoT Cities", "factuality (2c)", "1.000", "0.999", "−0.07% (饱和)"],
        ["MetaTool", "tool need (2c)", "0.998", "0.996", "−0.25% (饱和)"],
        ["RetrievalQA", "retrieval (2c)", "0.939", "0.946", "+0.66%"],
        ["common_claim", "factuality (3c)", "0.758", "0.776", "+1.88%"],
        ["E2H AMC 3c", "difficulty (3c)", "0.893", "0.914", "+2.06%"],
        ["E2H AMC 5c", "difficulty (5c)", "0.875", "0.898", "+2.28%"],
        ["When2Call", "routing (3c)", "0.874", "0.938", "+6.41%"],
        ["FAVA", "hallucination (2c)", "0.986", "0.991", "+0.51%"],
        ["RAGTruth", "hallucination (2c)", "0.881", "0.885", "+0.42%"],
    ])

add_text(s, 0.8, 5.8, 11.5, 0.4, "7/7 非饱和数据集全部提升  |  Wilcoxon p = 0.0098", 24, True, PRIMARY, PP_ALIGN.CENTER)
add_text(s, 0.8, 6.4, 11.5, 0.4, "两个loss是baseline已≈1.0的饱和数据集，无提升空间", 16, False, GRAY, PP_ALIGN.CENTER)

# ================================================================
# 9. When2Call deep dive
# ================================================================
s = add_slide()
title_box(s, "为什么When2Call能提升6.41%?")

bullets(s, 0.8, 1.3, 5, [
    ("When2Call: 3类tool routing", {"s": 22, "b": True, "c": ACCENT, "sb": 0}),
    ("不用工具 / 计算器 / 搜索引擎", {"s": 17, "c": GRAY}),
    ("", {"sb": 20}),
    ("Last-token (只看最后一个token):", {"s": 20}),
    ("  AUROC = 0.904", {"s": 24, "b": True, "sb": 6}),
    ("", {"sb": 16}),
    ("Mean-pool (看全部token的平均):", {"s": 20}),
    ("  AUROC = 0.933  (+2.9%)", {"s": 24, "b": True, "c": GREEN, "sb": 6}),
    ("", {"sb": 16}),
    ("Routing需要理解整个prompt的意图", {"s": 18, "c": GRAY}),
    ("不能只看最后一个token", {"s": 18, "c": GRAY}),
], size=20, spacing=6)

bullets(s, 7, 1.3, 5.5, [
    ("去掉某个view后AUROC下降多少:", {"s": 20, "b": True, "c": ACCENT, "sb": 0}),
    ("", {"sb": 16}),
    ("去掉mean-pool:        −2.26%", {"s": 22, "b": True, "c": RED}),
    ("去掉last-token:        −0.46%", {"s": 20, "sb": 12}),
    ("去掉gen mean-pool:    −0.17%", {"s": 20}),
    ("去掉head activation:  −0.12%", {"s": 20}),
    ("去掉probe methods:    −0.07%", {"s": 20}),
    ("", {"sb": 20}),
    ("Mean-pooled prompt hidden state", {"s": 20, "b": True, "c": PRIMARY}),
    ("= 最重要的互补信号", {"s": 20, "b": True, "c": PRIMARY}),
    ("之前没有probing方法系统地用过", {"s": 18, "c": GRAY}),
], size=20, spacing=6)

# ================================================================
# 10. Findings
# ================================================================
s = add_slide()
title_box(s, "Main Findings")

bullets(s, 0.8, 1.4, 11, [
    ("1. 不同probing方法的错误模式确实不同", {"s": 24, "b": True, "c": PRIMARY, "sb": 0}),
    ("简单线性融合即可显著提升 (7/7非饱和数据集, p=0.0098)", {"s": 19, "sb": 6}),
    ("", {"sb": 20}),
    ("2. Mean-pooled prompt表示是最强的互补信号", {"s": 24, "b": True, "c": PRIMARY}),
    ("现有方法全部用last-token，没有人系统用mean-pool", {"s": 19, "sb": 6}),
    ("在routing任务上比last-token强2.9%", {"s": 19}),
    ("", {"sb": 20}),
    ("3. 哪种view重要取决于任务类型", {"s": 24, "b": True, "c": PRIMARY}),
    ("Routing → mean-pool / Hallucination → probes+attn / Difficulty → hidden+heads", {"s": 19, "sb": 6}),
    ("", {"sb": 20}),
    ("4. 小数据集 (800-3500样本) 下线性模型最优", {"s": 24, "b": True, "c": PRIMARY}),
    ("Neural fusion (493K参数) 全部失败: −2% ~ −9%", {"s": 19, "sb": 6}),
], spacing=4)

# ================================================================
# 11. Next Steps
# ================================================================
s = add_slide()
title_box(s, "Limitations & Next Steps")

bullets(s, 0.8, 1.4, 5, [
    ("局限", {"s": 24, "b": True, "c": RED, "sb": 0}),
    ("", {"sb": 12}),
    ("• 只用了1个LLM (Qwen2.5-7B)", {"sb": 4}),
    ("  需要第2个模型验证", {"s": 17, "c": GRAY}),
    ("", {"sb": 12}),
    ("• 只做了分类任务", {}),
    ("  回归/multi-label未测", {"s": 17, "c": GRAY}),
    ("", {"sb": 12}),
    ("• 方法novelty一般", {}),
    ("  价值在findings不在架构", {"s": 17, "c": GRAY}),
], size=20, spacing=6)

bullets(s, 7, 1.4, 5.5, [
    ("下一步", {"s": 24, "b": True, "c": GREEN, "sb": 0}),
    ("", {"sb": 12}),
    ("1. 写论文 (数据和分析已ready)", {"sb": 4}),
    ("", {"sb": 12}),
    ("2. 跑第2个模型 (impact最大)", {}),
    ("   Llama-3-8B 或 Qwen2.5-3B", {"s": 17, "c": GRAY}),
    ("", {"sb": 12}),
    ("3. 更深入的分析", {}),
    ("   Layer-wise pattern, error cases", {"s": 17, "c": GRAY}),
    ("", {"sb": 16}),
    ("Target: NeurIPS 2026 (~May)", {"b": True, "c": ACCENT}),
], size=20, spacing=6)

# ================================================================
# 12. Thank You
# ================================================================
s = add_slide()
add_text(s, 1, 2.5, 11, 0.8, "Thank You / Questions?", 44, True, PRIMARY, PP_ALIGN.CENTER)
add_text(s, 1, 4.5, 11, 0.4, "12 baselines × 9 datasets × 11 views", 20, False, GRAY, PP_ALIGN.CENTER)
add_text(s, 1, 5.1, 11, 0.4, "7/7 wins  |  p = 0.0098  |  When2Call +6.41%", 20, False, GRAY, PP_ALIGN.CENTER)

prs.save("presentation.pptx")
print("Done: presentation.pptx (12 slides)")
