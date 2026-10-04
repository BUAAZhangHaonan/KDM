"""Place ImageGen artwork, original photographs and exact data in PDF pages.

The source photographs and generated PNG files are immutable inputs. The PDF
composition places the original photographs into reserved slots and draws the
quantitative ruler as PDF objects; no experimental image is regenerated.
Run with the bundled document Python (reportlab, Pillow).
"""
from pathlib import Path
import hashlib
import json
import os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, white

ROOT = Path(__file__).resolve().parents[1]
AS = ROOT / 'figure_sources/assets'
OUT = ROOT / 'figures'
WIDTH = 174 / 25.4 * 72
fonts = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
pdfmetrics.registerFont(TTFont('ArialExact', str(fonts/'arial.ttf')))
pdfmetrics.registerFont(TTFont('ArialExactBold', str(fonts/'arialbd.ttf')))

def page(name, art, width=WIDTH):
    w, h = Image.open(art).size
    height = width * h / w
    c = canvas.Canvas(str(OUT/f'{name}.pdf'), pagesize=(width, height))
    c.setTitle(name.replace('_', ' '))
    c.setAuthor('KDM: ImageGen artwork; original evidence and vector annotations')
    c.drawImage(str(art), 0, 0, width, height)
    return c, w, h, height

def original_photo(c, photo, rect, source_width, source_height, page_height):
    x0, y0, x1, y1 = rect  # ImageGen template pixels, origin at top left
    s = WIDTH / source_width
    pw, ph = Image.open(photo).size
    factor = min((x1-x0)*s/pw, (y1-y0)*s/ph)
    draw_w, draw_h = pw*factor, ph*factor
    x = x0*s + ((x1-x0)*s-draw_w)/2
    y = page_height-y1*s + ((y1-y0)*s-draw_h)/2
    c.drawImage(str(photo), x, y, draw_w, draw_h)

def main():
    inputs = [AS/'imagegen_architecture_print.png', AS/'imagegen_teaser_layout.png',
              AS/'beef_carpaccio_eval_039.jpg', AS/'apple_pie_eval_037.jpg']
    c, *_ = page('fig03_three_condition_duty_schematic', inputs[0], width=83/25.4*72)
    c.showPage(); c.save()
    c, w, h, H = page('fig01_authentic_case_teaser', inputs[1])
    original_photo(c, inputs[2], (60,138,480,538), w,h,H)
    original_photo(c, inputs[3], (1246,138,1600,538), w,h,H)
    # Exact data layer replaces the illustrative ruler in the reserved space.
    scale = WIDTH/w
    c.setFillColor(white); c.rect(0,0,1183*scale,217*scale,fill=1,stroke=0)
    left, right, y = 58*scale, 1120*scale, 112*scale
    xmin, xmax = -.30, .55
    X = lambda v: left+(v-xmin)/(xmax-xmin)*(right-left)
    c.setFillColor(HexColor('#F9ECE9'))
    c.rect(left,y-11,X(0)-left,22,fill=1,stroke=0)
    c.setFillColor(HexColor('#E8F4F1'))
    c.rect(X(0),y-11,right-X(0),22,fill=1,stroke=0)
    c.setStrokeColor(HexColor('#8293A3')); c.setLineWidth(.65)
    c.line(left,y,right,y)
    c.setStrokeColor(HexColor('#45576A')); c.line(X(0),y-11,X(0),y+11)
    c.setStrokeColor(HexColor('#258C83')); c.setLineWidth(1.6)
    c.line(X(-.1875),y,X(.421875),y)
    p=c.beginPath(); p.moveTo(X(.421875),y); p.lineTo(X(.421875)-5,y+3)
    p.moveTo(X(.421875),y); p.lineTo(X(.421875)-5,y-3); c.drawPath(p)
    for v, label, col in [(-.1875,'−0.1875','#C45A51'), (0,'0','#44556A'), (.421875,'+0.4219','#258C83')]:
        c.setFillColor(HexColor(col))
        if v: c.circle(X(v),y,2.4,fill=1,stroke=0)
        c.setFont('ArialExact',8); c.drawCentredString(X(v),y-22,label)
    c.setFillColor(HexColor('#19364C')); c.setFont('ArialExact',8)
    c.drawCentredString((left+right)/2,y+16,'Abstention margin at the first token')
    c.showPage(); c.save()
    manifest = {'tool':'built-in ImageGen', 'experimental_photos':'original JPEG pixels placed in PDF, aspect ratio preserved',
        'quantitative_ruler': {'domain':[-.3,.55], 'endpoints':[-.1875,.421875], 'zero':0},
        'architecture':'conceptual illustration; named g and S_g ports refer to upper outputs; no measured probabilities drawn',
        'prompts':['imagegen_architecture_print_prompt.txt','imagegen_architecture_ports_edit.txt','imagegen_teaser_prompt.txt'],
        'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (ROOT/'figure_sources/imagegen_provenance.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Composed teaser and architecture as PDF; original photos and numerical ruler preserved.')

if __name__ == '__main__':
    main()
