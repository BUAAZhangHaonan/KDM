"""Export this manuscript's main text and two tables from its canonical TeX."""
from pathlib import Path
import re

P = Path(__file__).resolve().parents[1]

def argument(text, start):
    assert text[start] == '{'
    depth = 1
    for end in range(start + 1, len(text)):
        if text[end] == '{' and text[end - 1] != '\\': depth += 1
        if text[end] == '}' and text[end - 1] != '\\': depth -= 1
        if not depth: return text[start + 1:end], end + 1
    raise ValueError('unclosed argument')

def replace_command(text, command, wrapper):
    token = '\\' + command + '{'
    while token in text:
        start = text.index(token)
        value, end = argument(text, start + len(token) - 1)
        text = text[:start] + wrapper(value) + text[end:]
    return text

aux = (P / 'paper_v3_zh.aux').read_text(encoding='utf-8')
labels = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}', aux))
citations = dict(re.findall(r'\\bibcite\{([^}]+)\}\{\{(\d+)\}', aux))

def inline(text):
    text = replace_command(text, 'textbf', lambda v: '**' + v + '**')
    text = replace_command(text, 'emph', lambda v: '*' + v + '*')
    text = re.sub(r'\\(?:eq)?ref\{([^}]+)\}', lambda m: labels[m[1]], text)
    text = re.sub(r'\\citep\{([^}]+)\}', lambda m: '[' + ', '.join(citations[k] for k in m[1].split(',')) + ']', text)
    text = re.sub(r'\\label\{[^}]+\}', '', text)
    return text.replace('\\%', '%').replace('\\&', '&').replace('\\_', '_').replace('~', ' ')

def table_markdown(name):
    source = (P / 'tables' / name).read_text(encoding='utf-8')
    caption = argument(source, source.index('\\caption{') + len('\\caption'))[0]
    blocks = []
    for body in re.findall(r'\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}', source, re.S):
        if name == 'main_joint.tex':
            headers = ['模型','方法','Food Acc','Food P','Food R','Food J','Viz Q','Viz P','Viz R','Viz J']
        elif '平均' in body:
            headers = ['实例','对照','平均 ΔJ','范围']
        else:
            headers = ['模型','表达','VCD 秒/题','IP-VCD 秒/题','CDA 秒/题']
        rows = []; current_model = ''
        for line in body.splitlines():
            if '&' not in line or '\\multicolumn' in line: continue
            if line.lstrip().startswith(('模型','实例','& & Acc')): continue
            line = re.sub(r'\\(?:midrule|bottomrule|toprule)', '', line)
            line = line.split('\\\\')[0]
            cells = [inline(c.strip()).replace('--', '—') for c in line.split('&')]
            if len(cells) != len(headers): continue
            if name == 'main_joint.tex':
                current_model = cells[0] or current_model; cells[0] = current_model
            rows.append('| ' + ' | '.join(cells) + ' |')
        blocks.append('| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + '\n'.join(rows))
    return inline(caption) + '\n\n' + '\n\n'.join(blocks)

source = (P / 'paper_v3_zh.tex').read_text(encoding='utf-8')
source = source[source.index('\\section*{摘要}'):source.index('\\clearpage\\nobalance')]
figures = {}
def take_figure(match):
    block = match[0]
    stem = re.search(r'figures/([^}]+)\.pdf', block)[1]
    caption = argument(block, block.index('\\caption{') + len('\\caption'))[0]
    key = re.search(r'\\label\{([^}]+)\}', block)[1]
    figures[key] = f'![图 {labels[key]}](figures/{stem}.png)\n\n图 {labels[key]}　{inline(caption)}\n\n'
    return ''
source = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', take_figure, source, flags=re.S)
source = re.sub(r'\\input\{tables/[^}]+\}', '', source)
source = re.sub(r'\\(?:FloatBarrier|newpage|balance|noindent)\b', '', source)
source = re.sub(r'\\section\*?\{([^}]+)\}', r'\n## \1\n', source)
source = re.sub(r'\\subsection\{([^}]+)\}', r'\n### \1\n', source)
source = source.replace('\\begin{equation}', '\n$$\n').replace('\\end{equation}', '\n$$\n')
source = inline(source)
placements = {
    '## 相关工作': figures['fig:teaser'] + '## 相关工作',
    '## 弃权怎样在视觉对比中流失': '## 弃权怎样在视觉对比中流失\n\n' + figures['fig:mechanism'],
    '## 指令保持的三条件解码': '## 指令保持的三条件解码\n\n' + figures['fig:architecture'],
    '### 视觉纠错与谨慎作答共同提高收益': table_markdown('main_joint.tex') + '\n\n' + figures['fig:food'] + '### 视觉纠错与谨慎作答共同提高收益',
    '### 真实不可回答问题上的改善更一致': '### 真实不可回答问题上的改善更一致\n\n' + figures['fig:viz'],
    '## 结构消融与计算开销': '## 结构消融与计算开销\n\n' + table_markdown('ablation_cost.tex'),
}
for old,new in placements.items():
    assert source.count(old) == 1, old
    source = source.replace(old,new)
source = re.sub(r'\n{3,}', '\n\n', source).strip()
title = '# Lost in Contrast: Preserving Abstention in Multimodal Contrastive Decoding\n\n保留弃权指令的多模态对比解码 · v3\n\n'
footer = '\n\n## 参考文献与附录\n\n参考文献编号与 [PDF](paper_v3_zh.pdf) 一致，条目见 [BibTeX](refs/references.bib)。\n\n- [完整九模型表、理论与实验附录](appendix_v3_zh.md)\n- [开发选择、更新基线、CDA 与重放补充](appendix_supplement_v3.md)\n- [图表原始汇总与来源](statistics/SOURCE_MAPPING.md)\n'
(P / 'paper_v3_zh.md').write_text(title + source + footer,encoding='utf-8')
assert len(figures) == 5
print('Synchronized Markdown: 5 figures, 2 table groups, canonical main text.')
