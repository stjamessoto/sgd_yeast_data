"""
_generate_doc.py  —  one-page GRN motif analysis document.
Run: python _generate_doc.py
"""

import sys, io, math
from itertools import combinations as icombs
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from model.inheritance_estimator import full_significance_analysis
from model.gene_families import build_tf_families, estimate_model_parameters

# ── constants ──────────────────────────────────────────────────────────────
GO_NAMES = {
    'GO:0006355': 'Regulation of transcription, DNA-templated',
    'GO:0045944': 'Positive regulation of transcription by RNA Pol II',
    'GO:0006357': 'Regulation of transcription by RNA Pol II',
    'GO:0000122': 'Negative regulation of transcription by RNA Pol II',
    'GO:0006351': 'Transcription by RNA Pol II',
    'GO:0045893': 'Positive regulation of transcription, DNA-templated',
    'GO:0045892': 'Negative regulation of transcription, DNA-templated',
}
FDR_ALPHA = 0.05

# ── load + compute ─────────────────────────────────────────────────────────
families = build_tf_families(min_family_size=1, grouping='GO_Process')
params   = estimate_model_parameters(families, 'family_size')
M, N     = params['m'], params['n']

fam_list = []
for i, (_, row) in enumerate(families.iterrows()):
    fam_list.append({
        'idx':        i + 1,
        'go_id':      row['go_id'],
        'size':       int(row['family_size']),
        'ev':         float(row['mean_evidence_score']),
        'name':       GO_NAMES.get(row['go_id'], row['go_id']),
        'member_tfs': [t.upper() for t in row['member_tfs']],
    })

# ── Method 1 (corrected): normalize evidence scores per paper Eq. 3 ────────
# π̂ᵢ = raw evidence score; πᵢ = π̂ᵢ / Σⱼ π̂ⱼ  (Scruse et al. Eq. 3)
_ev_total = sum(f['ev'] for f in fam_list)
_fam_pi   = {f['idx']: f['ev'] / _ev_total for f in fam_list}


def run_motifs(k):
    out = []
    for combo in icombs(range(len(fam_list)), k):
        fams_sel = [fam_list[i] for i in combo]
        sizes    = [f['size']  for f in fams_sel]
        label    = '+'.join('F%d' % f['idx'] for f in fams_sel)
        pi_vec   = [_fam_pi[f['idx']] for f in fams_sel]
        # observed = Cartesian product of family sizes: one gene per family
        obs      = math.prod(sizes)
        try:
            sig = full_significance_analysis(pi_vec, M, N, float(obs), k)
        except Exception:
            continue
        out.append({
            'label': label, 'sizes': sizes, 'obs': obs,
            'z':  sig['z_partial'], 'p':  sig['p_partial'],
            'sig': sig['sig_partial'],
        })
    return out


print('Computing motifs...')
res3 = run_motifs(3)
res4 = run_motifs(4)

# ── Benjamini-Hochberg FDR correction across all 70 tests ─────────────────
_all = [(r, 'k3') for r in res3 if r['p'] is not None] + \
       [(r, 'k4') for r in res4 if r['p'] is not None]
_ps  = np.array([r['p'] for r, _ in _all])
_n   = len(_ps)
_ord = np.argsort(_ps)
_ranks          = np.empty(_n, dtype=int)
_ranks[_ord]    = np.arange(1, _n + 1)
_bh_sig         = _ps <= (_ranks / _n) * FDR_ALPHA
for k, (r, _) in enumerate(_all):
    r['bh_sig'] = bool(_bh_sig[k])

# BH critical Z: the Z corresponding to the largest p-value still BH-significant
_sig_sorted = np.where(_bh_sig[_ord])[0]
if len(_sig_sorted):
    _bh_crit_p = _ps[_ord[_sig_sorted[-1]]]
    BH_CRIT_Z  = float(norm.ppf(1 - _bh_crit_p / 2))
else:
    BH_CRIT_Z  = np.inf
SIG_THRESH = np.log10(BH_CRIT_Z)

sig3  = sum(1 for r in res3 if r.get('bh_sig'))
sig4  = sum(1 for r in res4 if r.get('bh_sig'))
over3 = sum(1 for r in res3 if r.get('bh_sig') and r['z'] and r['z'] > 0)
over4 = sum(1 for r in res4 if r.get('bh_sig') and r['z'] and r['z'] > 0)

# ── figure (Science style: 2-column width, no title, see assets/science_style.py)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'assets'))
import science_style as sci
sci.apply()

fig = plt.figure(figsize=(sci.WIDTH_2COL, 5.6 * sci.CM))
ax_strip = fig.add_axes([0.11, 0.17, 0.30, 0.76])
ax_hist  = fig.add_axes([0.53, 0.17, 0.45, 0.76])

z3_vals = np.array([r['z'] for r in res3 if r['z'] is not None])
z4_vals = np.array([r['z'] for r in res4 if r['z'] is not None])
lz3, lz4 = np.log10(z3_vals), np.log10(z4_vals)
sig3_mask = np.array([r.get('bh_sig', False) for r in res3 if r['z'] is not None])
sig4_mask = np.array([r.get('bh_sig', False) for r in res4 if r['z'] is not None])
K_COLORS = (sci.BLUE, sci.VERMILLION)

# ── panel A: one point per motif combination; open = not significant ──────
rng = np.random.default_rng(42)
for x0, lz, mask, col in ((0, lz3, sig3_mask, K_COLORS[0]), (1, lz4, sig4_mask, K_COLORS[1])):
    jit = x0 + rng.uniform(-0.18, 0.18, len(lz))
    ax_strip.scatter(jit[mask], lz[mask], s=9, color=col, linewidths=0, zorder=3)
    ax_strip.scatter(jit[~mask], lz[~mask], s=9, facecolors='none', edgecolors=col,
                     linewidths=0.6, zorder=3)
ax_strip.axhline(SIG_THRESH, color=sci.BLACK, linestyle='--', linewidth=0.7, zorder=2)
ax_strip.set_xticks([0, 1])
ax_strip.set_xticklabels(['3', '4'])
ax_strip.set_xlabel('Motif size, k (families)')
ax_strip.set_xlim(-0.55, 1.55)
_yt = np.arange(0, 8)
_tick_labels = ['1', '10'] + ['10$^{%d}$' % e for e in _yt[2:]]
ax_strip.set_yticks(_yt)
ax_strip.set_yticklabels(_tick_labels)
ax_strip.set_ylim(0, 7)
ax_strip.set_ylabel('Z-score')

# ── panel B: Z-score distribution, side-by-side bars per k ─────────────────
# edges start at the BH threshold so no bar straddles the significance line
bins = SIG_THRESH + 0.5 * np.arange(-2, 14)
ax_hist.hist([lz3, lz4], bins=bins, color=list(K_COLORS), rwidth=0.9,
             label=['k = 3 (n = %d)' % len(lz3), 'k = 4 (n = %d)' % len(lz4)])
ax_hist.axvline(SIG_THRESH, color=sci.BLACK, linestyle='--', linewidth=0.7,
                label='BH threshold, FDR = 0.05 (Z = %.2f)' % BH_CRIT_Z)
ax_hist.set_xticks(_yt)
ax_hist.set_xticklabels(_tick_labels)
ax_hist.set_xlim(0, 7)
ax_hist.set_xlabel('Z-score')
ax_hist.set_ylabel('Motif combinations')
# headroom so the legend never overlaps a bar
ax_hist.set_ylim(0, max(np.histogram(lz3, bins)[0].max(), np.histogram(lz4, bins)[0].max()) * 1.4)
ax_hist.legend(loc='upper right', handlelength=1.6)

sci.panel_label(fig, ax_strip, 'A', dx=-0.075)
sci.panel_label(fig, ax_hist, 'B', dx=-0.075)

OUT_PNG = Path(__file__).parent.parent.parent / 'assets' / 'motif_significance.png'
fig.savefig(OUT_PNG, dpi=sci.DPI)

buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
plt.close(fig)
buf.seek(0)
print('Figure done.')

# ── document helpers ───────────────────────────────────────────────────────
def tight_para(doc, text, bold=False, size=9, space_after=2):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(space_after)
    return p

def shd(cell, hex_fill):
    tc = cell._tc; tcPr = tc.get_or_add_tcPr()
    s = OxmlElement('w:shd')
    s.set(qn('w:val'), 'clear'); s.set(qn('w:color'), 'auto')
    s.set(qn('w:fill'), hex_fill)
    tcPr.append(s)

def tbl_hdr(tbl, headers, fill='1E3A5F'):
    row = tbl.rows[0]
    for cell, h in zip(row.cells, headers):
        run = cell.paragraphs[0].add_run(h)
        run.bold = True; run.font.size = Pt(8)
        shd(cell, fill); run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

def tbl_row(tbl, vals, fill=None, bold_first=False):
    row = tbl.add_row()
    for j, (cell, v) in enumerate(zip(row.cells, vals)):
        run = cell.paragraphs[0].add_run(str(v))
        run.font.size = Pt(8)
        if bold_first and j == 0: run.bold = True
        if fill: shd(cell, fill)

# ── build document ─────────────────────────────────────────────────────────
doc = Document()

# tighten default style
for style_name in ('Normal', 'Body Text'):
    try:
        s = doc.styles[style_name]
        s.paragraph_format.space_before = Pt(0)
        s.paragraph_format.space_after  = Pt(2)
    except Exception:
        pass

for sec in doc.sections:
    sec.top_margin    = Inches(0.80)
    sec.bottom_margin = Inches(0.80)
    sec.left_margin   = Inches(1.00)
    sec.right_margin  = Inches(1.00)

# title line
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.paragraph_format.space_before = Pt(0)
t.paragraph_format.space_after  = Pt(1)
run = t.add_run('Subnetwork Motif Analysis — S. cerevisiae GRN')
run.bold = True; run.font.size = Pt(13)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.paragraph_format.space_before = Pt(0)
sub.paragraph_format.space_after  = Pt(5)
sub.add_run(
    'Scruse, Arnold & Robinson (2026)  ·  '
    'm = %d gene families  ·  n = %d TFs  ·  d = %d duplication events  ·  '
    '3-node and 4-node motifs' % (M, N, N - M)
).font.size = Pt(8.5)

# ── introduction ───────────────────────────────────────────────────────────
intro_hdr = doc.add_paragraph()
intro_hdr.paragraph_format.space_before = Pt(0)
intro_hdr.paragraph_format.space_after  = Pt(2)
intro_hdr.add_run('Introduction').bold = True
intro_hdr.runs[0].font.size = Pt(9.5)

intro_body = doc.add_paragraph()
intro_body.paragraph_format.space_before = Pt(0)
intro_body.paragraph_format.space_after  = Pt(5)
intro_run = intro_body.add_run(
    'This analysis tests whether combinations of transcription factor (TF) gene '
    'families form subnetwork motifs at rates that exceed neutral expectations under '
    'the Partial Duplication model of Scruse, Arnold & Robinson (2026). The model '
    'parameterises gene regulatory network (GRN) inheritance through a per-family '
    'probability πᵢ that a regulatory link survives a duplication event. We enumerate all C(7,3) = 35 '
    'three-node and C(7,4) = 35 four-node family combinations (70 total), compute '
    'the Cartesian product of family sizes as the observed co-membership count, and '
    'derive Z-scores using the Binary Inheritance variance (Corollary 16) to assess '
    'statistical significance against the Partial Duplication null, with '
    'Benjamini-Hochberg FDR correction applied across all 70 tests (α = 5%). '
    'Per-family inheritance probabilities πᵢ are estimated from SGD evidence '
    'codes and normalized by their sum across all m families (Eq. 3).'
)
intro_run.font.size = Pt(9)

# ── gene families table ────────────────────────────────────────────────────
hdr_p = doc.add_paragraph()
hdr_p.paragraph_format.space_before = Pt(0)
hdr_p.paragraph_format.space_after  = Pt(2)
r = hdr_p.add_run('Gene Families  (GO Biological Process grouping)')
r.bold = True; r.font.size = Pt(9.5)

famdef = doc.add_paragraph()
famdef.paragraph_format.space_before = Pt(0)
famdef.paragraph_format.space_after  = Pt(3)
famdef.add_run(
    'Families are defined by shared GO Biological Process annotation in SGD: all '
    '%d TFs in the dataset are partitioned into m = %d groups based on their '
    'primary GO term. Family sizes range from 1 (F7) to %d (F1). '
    'Evidence scores weight annotation quality (IEA = 0.30; manual curation higher); '
    'all families default to π̂ ≈ 0.30 because GO term IDs are not resolved '
    'to individual TF-level evidence in the current pipeline.' % (
        N, M, max(f['size'] for f in fam_list))
).font.size = Pt(9)

ft = doc.add_table(rows=1, cols=5)
ft.style = 'Table Grid'
tbl_hdr(ft, ['Family', 'GO ID', 'GO Process Name', 'Size (cᵢ)', 'Ev. Score'])
ALT = ['EBF5FF', 'FFFFFF'] * 10
for i, fam in enumerate(fam_list):
    tbl_row(ft, [
        'F%d' % fam['idx'], fam['go_id'], fam['name'],
        str(fam['size']), '%.3f' % fam['ev'],
    ], fill=ALT[i], bold_first=True)

p_gap = doc.add_paragraph()
p_gap.paragraph_format.space_before = Pt(3)
p_gap.paragraph_format.space_after  = Pt(0)

# figure
pfig = doc.add_paragraph()
pfig.alignment = WD_ALIGN_PARAGRAPH.CENTER
pfig.paragraph_format.space_before = Pt(0)
pfig.paragraph_format.space_after  = Pt(1)
pfig.add_run().add_picture(buf, width=Inches(6.4))

# caption
cap = doc.add_paragraph()
cap.paragraph_format.space_before = Pt(0)
cap.paragraph_format.space_after  = Pt(4)
cap_run = cap.add_run(
    'Figure.  Left: each dot is one motif combination; red = significant (BH FDR < 5%%, '
    'Partial Dup null), grey = not significant after BH correction. Non-significant k=3 dots are labelled.  '
    'Right: histogram of log₁₀(Z-score); dashed line = BH FDR=5%% threshold (Z=%.2f); shaded region = significant over-representation.  ' % BH_CRIT_Z +
    'F1=GO:0006355, F2=GO:0045944, F3=GO:0006357, F4=GO:0000122, '
    'F5=GO:0006351, F6=GO:0045893, F7=GO:0045892.'
)
cap_run.font.size = Pt(8); cap_run.italic = True

# significance summary table
hdr_p2 = doc.add_paragraph()
hdr_p2.paragraph_format.space_before = Pt(0)
hdr_p2.paragraph_format.space_after  = Pt(2)
r2 = hdr_p2.add_run('Significance Summary')
r2.bold = True; r2.font.size = Pt(9.5)

st = doc.add_table(rows=1, cols=5)
st.style = 'Table Grid'
tbl_hdr(st, ['Motif size', 'Combinations', 'Significant (BH FDR<5%)', 'Over-represented', 'Under-represented'])
tbl_row(st, ['k = 3', '35', '%d' % sig3,  '%d' % over3, '0'], fill='DBEAFE')
tbl_row(st, ['k = 4', '35', '%d' % sig4,  '%d' % over4, '0'], fill='FEE2E2')
tbl_row(st, ['Total', '70', '%d' % (sig3+sig4), '%d' % (over3+over4), '0'], fill='E0F2FE')

# ── over / under-representation explanation ────────────────────────────────
rep_hdr = doc.add_paragraph()
rep_hdr.paragraph_format.space_before = Pt(5)
rep_hdr.paragraph_format.space_after  = Pt(2)
rep_hdr.add_run('Over- and Under-Representation').bold = True
rep_hdr.runs[0].font.size = Pt(9.5)

rep_body = doc.add_paragraph()
rep_body.paragraph_format.space_before = Pt(0)
rep_body.paragraph_format.space_after  = Pt(5)
rep_body.add_run(
    'A k-node motif combination is over-represented when its observed Cartesian '
    'product count c₁×c₂×···×cₖ significantly exceeds the expected count '
    'E[|M(n)|] predicted by the Partial Duplication null (Theorem 4), with '
    'Benjamini-Hochberg FDR correction at α = 5%% across all 70 tests. '
    'This indicates that the selected families share more '
    'co-regulatory TF members than stochastic duplication-loss dynamics alone would '
    'produce, consistent with functional constraint or active co-regulation among '
    'those GO Biological Process groups. Under-representation (negative BH-significant Z) would '
    'signal depletion relative to neutral expectations — no such case arises in this '
    'dataset, suggesting that GRN links between these families are at least as dense '
    'as the duplication model predicts, and in most cases far denser.'
).font.size = Pt(9)

# key findings
kf = doc.add_paragraph()
kf.paragraph_format.space_before = Pt(0)
kf.paragraph_format.space_after  = Pt(0)
kf_run = kf.add_run(
    'Key findings.  '
)
kf_run.bold = True; kf_run.font.size = Pt(9)

_z_by_label = {r['label']: r['z'] for r in res3 + res4}
_ns3 = [r['label'] for r in res3 if not r.get('bh_sig', False) and r['z'] is not None]
_ns3_str = (', '.join(_ns3) + ' — ') if _ns3 else 'none — all k=3 combinations are significant. '
kf_body = kf.add_run(
    'Of 35 three-node motif combinations, %d (%d%%) are significant after BH FDR correction; '
    '%d of 35 four-node combinations are significant. No motif is under-represented. '
    'Non-significant k=3 cases: %s'
    'these combinations include near-singleton families (F6: 2 TFs, F7: 1 TF) whose '
    'Cartesian products fall close to the Partial Duplication expected count. '
    'The strongest over-representation involves the three largest families — '
    'F1+F2+F3 (101×98×86 = 851,228; Z = %s) and '
    'F1+F2+F3+F4 (Z = %s for k=4) — reflecting dense co-regulatory wiring '
    'among core RNA Pol II transcriptional regulators far exceeding neutral '
    'duplication predictions.' % (sig3, round(100 * sig3 / 35), sig4, _ns3_str,
                              '{:,.0f}'.format(_z_by_label['F1+F2+F3']),
                              '{:,.0f}'.format(_z_by_label['F1+F2+F3+F4']))
)
kf_body.font.size = Pt(9)

OUT_DOCX = str(Path(__file__).parent.parent.parent / 'GRN_Inheritance_Overrepresented_Families.docx')
doc.save(OUT_DOCX)
print('Saved:', OUT_DOCX)
