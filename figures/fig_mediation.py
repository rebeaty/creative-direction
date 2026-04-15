"""
Mediation path diagrams: g/c -> CD -> Creativity (Sample 2).
Bootstrap CIs (10,000 resamples, bias-corrected).
Output: figures/figure_mediation.{svg,png}
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import bootstrap_env  # noqa: F401

sys.stdout.reconfigure(encoding='utf-8')

basedir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(basedir, 'scripts'))
from analysis_common import SEED, ensure_figure_dir, format_stars, get_sample2_mediation_results

outdir = ensure_figure_dir()

W, H = 900, 250
PANEL_W = 410
GAP = 80
PX = [0, PANEL_W + GAP]

NW, NH = 155, 48
FONT = 'Helvetica, Arial, sans-serif'
FONT_SIZE_NODE = 14
FONT_SIZE_COEFF = 13
FONT_SIZE_INDIRECT = 12
FONT_SIZE_PANEL = 16

C_BLACK = '#1a1a1a'
C_DARK = '#333333'
C_MID = '#888888'
C_STROKE = '#333333'
C_NODE_FILL = '#ffffff'
C_ACCENT = '#c0392b'

LW_PATH = 1.6
LW_DIRECT = 1.2
LW_NODE = 1.4

mediation = get_sample2_mediation_results(n_boot=10000, seed=SEED)

def fmt_coeff(value, p_value):
    return f"{value:.2f}".replace('0.', '.') + format_stars(p_value)

panels = [
    {
        'label': 'A',
        'iv': 'Intelligence (g)',
        'a': fmt_coeff(mediation['g']['a'], mediation['g']['a_p']),
        'b': fmt_coeff(mediation['g']['b'], mediation['g']['b_p']),
        'cp': fmt_coeff(mediation['g']['cp'], mediation['g']['cp_p']),
        'indirect': f"{mediation['g']['indirect']:.3f}".replace('0.', '.'),
        'ci': f"{mediation['g']['ci_lo']:.3f}, {mediation['g']['ci_hi']:.3f}".replace('0.', '.'),
    },
    {
        'label': 'B',
        'iv': 'Creative Ability (c)',
        'a': fmt_coeff(mediation['c']['a'], mediation['c']['a_p']),
        'b': fmt_coeff(mediation['c']['b'], mediation['c']['b_p']),
        'cp': fmt_coeff(mediation['c']['cp'], mediation['c']['cp_p']),
        'indirect': f"{mediation['c']['indirect']:.3f}".replace('0.', '.'),
        'ci': f"{mediation['c']['ci_lo']:.3f}, {mediation['c']['ci_hi']:.3f}".replace('0.', '.'),
    },
]


def marker(id_name, color, w=6, h=4.5):
    return (
        f'<marker id="{id_name}" viewBox="0 0 {w} {h}" '
        f'refX="{w}" refY="{h/2}" '
        f'markerWidth="{w}" markerHeight="{h}" orient="auto">'
        f'<path d="M0,0 L{w},{h/2} L0,{h}" fill="none" '
        f'stroke="{color}" stroke-width="0.7" stroke-linejoin="round" />'
        f'</marker>'
    )


def node_box(x, y, text, w=NW, h=NH):
    rx, ry = x - w / 2, y - h / 2
    return (
        f'<rect x="{rx}" y="{ry}" width="{w}" height="{h}" '
        f'fill="{C_NODE_FILL}" stroke="{C_STROKE}" stroke-width="{LW_NODE}" />\n'
        f'  <text x="{x}" y="{y + 1}" text-anchor="middle" '
        f'dominant-baseline="central" font-size="{FONT_SIZE_NODE}" '
        f'fill="{C_BLACK}" font-family="{FONT}">{text}</text>'
    )


def edge_pt(cx, cy, tx, ty, w=NW, h=NH, pad=1.5):
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    angle = math.atan2(dy, dx)
    hw, hh = w / 2, h / 2
    if abs(math.cos(angle)) * hh > abs(math.sin(angle)) * hw:
        sx = hw if dx > 0 else -hw
        sy = sx * math.tan(angle)
    else:
        sy = hh if dy > 0 else -hh
        sx = sy / math.tan(angle) if math.tan(angle) != 0 else 0
    return cx + sx + pad * math.cos(angle), cy + sy + pad * math.sin(angle)


def line(x1, y1, x2, y2, mid, color=C_DARK, width=LW_PATH, dash=''):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" fill="none"{dash_attr} '
        f'marker-end="url(#{mid})" />'
    )


def coeff(x, y, text, color=C_BLACK, italic=True):
    style = ' font-style="italic"' if italic else ''
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" '
        f'dominant-baseline="central" font-size="{FONT_SIZE_COEFF}" '
        f'fill="{color}" font-family="{FONT}"{style} '
        f'paint-order="stroke" stroke="white" stroke-width="3" '
        f'stroke-linejoin="round">{text}</text>'
    )


def draw_panel(px, data):
    iv_x, iv_y = px + 80, 165
    med_x, med_y = px + 205, 55
    dv_x, dv_y = px + 330, 165

    a1x, a1y = edge_pt(iv_x, iv_y, med_x, med_y)
    a2x, a2y = edge_pt(med_x, med_y, iv_x, iv_y)
    b1x, b1y = edge_pt(med_x, med_y, dv_x, dv_y)
    b2x, b2y = edge_pt(dv_x, dv_y, med_x, med_y)
    cp1x, cp1y = edge_pt(iv_x, iv_y, dv_x, dv_y)
    cp2x, cp2y = edge_pt(dv_x, dv_y, iv_x, iv_y)

    parts = []
    parts.append(line(a1x, a1y, a2x, a2y, 'ah'))
    parts.append(line(b1x, b1y, b2x, b2y, 'ah'))
    parts.append(line(cp1x, cp1y, cp2x, cp2y, 'ahd', color=C_MID, width=LW_DIRECT, dash='5,3'))
    parts.append(node_box(iv_x, iv_y, data['iv']))
    parts.append(node_box(med_x, med_y, 'Creative Direction'))
    parts.append(node_box(dv_x, dv_y, 'Product Creativity'))

    amx = (a1x + a2x) / 2 - 12
    amy = (a1y + a2y) / 2 - 4
    parts.append(coeff(amx, amy, data['a']))

    bmx = (b1x + b2x) / 2 + 12
    bmy = (b1y + b2y) / 2 - 4
    parts.append(coeff(bmx, bmy, data['b']))

    cmx = (cp1x + cp2x) / 2
    cmy = iv_y + 24
    parts.append(coeff(cmx, cmy, f"c′ = {data['cp']}", color=C_MID, italic=False))

    parts.append(
        f'<text x="{amx}" y="{amy - 12}" text-anchor="middle" dominant-baseline="central" '
        f'font-size="8" fill="{C_MID}" font-family="{FONT}" font-style="italic" '
        f'paint-order="stroke" stroke="white" stroke-width="3" stroke-linejoin="round">a</text>'
    )
    parts.append(
        f'<text x="{bmx}" y="{bmy - 12}" text-anchor="middle" dominant-baseline="central" '
        f'font-size="8" fill="{C_MID}" font-family="{FONT}" font-style="italic" '
        f'paint-order="stroke" stroke="white" stroke-width="3" stroke-linejoin="round">b</text>'
    )

    ind_text = f"ab = {data['indirect']}, 95% CI [{data['ci']}]"
    parts.append(
        f'<text x="{med_x}" y="{med_y - 28}" text-anchor="middle" dominant-baseline="central" '
        f'font-size="{FONT_SIZE_INDIRECT}" fill="{C_ACCENT}" font-family="{FONT}" font-style="italic">'
        f'{ind_text}</text>'
    )
    parts.append(
        f'<text x="{px + 6}" y="18" font-size="{FONT_SIZE_PANEL}" font-weight="bold" '
        f'fill="{C_BLACK}" font-family="{FONT}">{data["label"]}</text>'
    )
    return '\n  '.join(parts)


svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
    '<defs>',
    marker('ah', C_DARK),
    marker('ahd', C_MID),
    '</defs>',
    f'<rect width="{W}" height="{H}" fill="white" />',
]
for i, panel in enumerate(panels):
    svg.append(draw_panel(PX[i], panel))
svg.append('</svg>')
svg_content = '\n'.join(svg)

svg_path = os.path.join(outdir, 'figure_mediation.svg')
with open(svg_path, 'w', encoding='utf-8') as handle:
    handle.write(svg_content)
print(f'Saved: {svg_path}')
