"""
_get_estimates.py — print all k=3 and k=4 motif significance results to console.
Run: python _get_estimates.py
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from itertools import combinations as icombs

from model.inheritance_estimator import estimate_pi_from_evidence, full_significance_analysis
from model.gene_families import build_tf_families, estimate_model_parameters

families = build_tf_families(min_family_size=1, grouping='GO_Process')
params = estimate_model_parameters(families, 'family_size')
m, n = params['m'], params['n']

GO_NAMES = {
    'GO:0006355': 'Regulation of transcription, DNA-templated',
    'GO:0045944': 'Positive regulation of transcription by RNA Pol II',
    'GO:0006357': 'Regulation of transcription by RNA Pol II',
    'GO:0000122': 'Negative regulation of transcription by RNA Pol II',
    'GO:0006351': 'Transcription by RNA Pol II',
    'GO:0045893': 'Positive regulation of transcription, DNA-templated',
    'GO:0045892': 'Negative regulation of transcription, DNA-templated',
}

print("=" * 80)
print("GENE FAMILIES  m=%d  n=%d  d=%d duplication events" % (m, n, n - m))
print("=" * 80)
fam_list = []
for i, (_, row) in enumerate(families.iterrows()):
    fam_list.append({
        'idx': i + 1,
        'go_id': row['go_id'],
        'size': int(row['family_size']),
        'ev': float(row['mean_evidence_score']),
        'name': GO_NAMES.get(row['go_id'], row['go_id']),
    })
    print("F%d  %-12s  size=%3d  ev=%.4f  %s" % (
        i + 1, row['go_id'], int(row['family_size']),
        float(row['mean_evidence_score']),
        GO_NAMES.get(row['go_id'], row['go_id'])
    ))

num_fams = len(fam_list)


def run_motifs(k):
    results = []
    for combo in icombs(range(num_fams), k):
        fams_sel = [fam_list[i] for i in combo]
        go_ids   = [f['go_id'] for f in fams_sel]
        sizes    = [f['size']  for f in fams_sel]
        obs      = math.prod(sizes)
        label    = '+'.join('F%d' % f['idx'] for f in fams_sel)

        pi_res = estimate_pi_from_evidence(go_ids)
        pi_vec = pi_res['pi_vec']

        try:
            sig = full_significance_analysis(pi_vec, m, n, float(obs), k)
        except Exception as e:
            print("  ERROR %s: %s" % (label, e))
            continue

        results.append({
            'label':      label,
            'go_ids':     go_ids,
            'sizes':      sizes,
            'obs':        obs,
            'pi_hat':     sig['pi_hat'],
            'z_full':     sig['z_full'],
            'z_partial':  sig['z_partial'],
            'p_full':     sig['p_full'],
            'p_partial':  sig['p_partial'],
            'sig_full':   sig['sig_full'],
            'sig_partial': sig['sig_partial'],
            'e_full':     sig['expected_full'],
            'e_partial':  sig['expected_partial'],
        })
    return results


for k in [3, 4]:
    results = run_motifs(k)
    valid_z  = [r for r in results if r['z_partial'] is not None]
    sig_part = [r for r in valid_z if r['sig_partial'] != 'ns']
    over     = [r for r in valid_z if r['z_partial'] >  1.96]
    under    = [r for r in valid_z if r['z_partial'] < -1.96]

    print("")
    print("=" * 80)
    print("k=%d MOTIFS   %d combinations   sig(PartialDup)=%d  over=%d  under=%d" % (
        k, len(results), len(sig_part), len(over), len(under)))
    print("  Z range: min=%.3f  max=%.3f  (NaN=%d)" % (
        min((r['z_partial'] for r in valid_z), default=0),
        max((r['z_partial'] for r in valid_z), default=0),
        len(results) - len(valid_z),
    ))
    print("=" * 80)
    print("%-12s  %12s  %7s  %10s  %10s  %8s  %5s  %5s" % (
        "Combo", "Obs", "pi_hat", "E_full", "E_part", "Z_part", "Full", "Part"))
    for r in sorted(results, key=lambda x: x['z_partial'] or 0, reverse=True):
        zp = r['z_partial']
        print("%-12s  %12d  %7.4f  %10.2f  %10.4f  %8.3f  %5s  %5s" % (
            r['label'], r['obs'], r['pi_hat'],
            r['e_full'], r['e_partial'],
            zp if zp is not None else float('nan'),
            r['sig_full'], r['sig_partial']
        ))
