"""
ORA from SHAP (new downstream script)

- Reads Pan-cancer_mRNA.csv to obtain authoritative 3,217 mRNA background
- Reads shap_top.csv (produced by interpret_xgboost.py) and computes mean |SHAP| ranking for mRNA features
- Produces ranked_mrna_shap.csv and top_N gene lists
- Parses human cell marker file (human_cell_marker_/human_cell_marker.txt) provided in project directory
- Runs hypergeometric ORA for top N = 100,150,200 (configurable)
- Applies Benjamini-Hochberg FDR and saves raw/adj/significant tables and dotplots into ORA_results/

Usage:
    python ora_from_shap.py --top 100,150,200

Note (repo integration): this script expects `Pan-cancer_mRNA.csv` and
`shap_top20.csv` in the current working directory. See README "Execution".
"""
import os
import argparse
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def parse_cellmarker(path):
    # expects a tab-delimited file with header including 'cell_name' and 'symbol'
    df = pd.read_csv(path, sep='\t', dtype=str)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    if 'symbol' not in cols.values() and 'gene_symbol' not in cols.values():
        raise ValueError('CellMarker file does not contain a gene symbol column')
    # prefer 'cell_name' grouping
    if 'cell_name' in cols:
        cell_col = cols['cell_name']
    elif 'cell' in cols:
        cell_col = cols['cell']
    else:
        # fallback to tissue_type
        cell_col = cols.get('tissue_type', list(cols.values())[0])
    symbol_col = cols.get('symbol', cols.get('gene_symbol', 'symbol'))
    gs = defaultdict(set)
    for _, row in df.iterrows():
        cell = str(row[cell_col]) if pd.notna(row[cell_col]) else ''
        sym = str(row[symbol_col]) if pd.notna(row[symbol_col]) else ''
        if cell and sym and sym.upper() != 'NA':
            gs[cell].add(sym)
    return {k: sorted(v) for k, v in gs.items()}


def hypergeom_pval(N, K, n, k):
    # compute P(X >= k) for hypergeometric(N, K, n)
    if k > min(n, K):
        return 1.0
    denom = math.comb(N, n)
    s = 0
    for i in range(k, min(n, K) + 1):
        s += math.comb(K, i) * math.comb(N - K, n - i)
    return s / denom if denom > 0 else 1.0


def bh_adjust(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n, dtype=float)
    for rank, idx in enumerate(order, start=1):
        adjusted[idx] = pvals[idx] * n / rank
    # enforce monotonicity
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])
    adjusted = np.minimum(adjusted, 1.0)
    return adjusted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', default='150', help='Comma-separated top-N values, default 150')
    ap.add_argument('--cellmarker', default=os.path.join('human_cell_marker_', 'human_cell_marker.txt'), help='Path to human cell marker TSV provided in repository')
    ap.add_argument('--outdir', default='ORA_results', help='Output directory for ORA results and plots')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1. load authoritative background mRNA genes
    data_dir = os.path.abspath('.')
    mRNA_path = os.path.join(data_dir, 'Pan-cancer_mRNA.csv')
    if not os.path.exists(mRNA_path):
        raise FileNotFoundError(f'{mRNA_path} not found in working directory')
    mdf = pd.read_csv(mRNA_path, index_col=0)
    background_genes = [str(g) for g in mdf.index.tolist()]
    N = len(background_genes)
    print('Background mRNA genes loaded:', N)

    # 2. load SHAP ranking produced by interpret_xgboost.py
    shap_path = os.path.join(data_dir, 'shap_top20.csv')
    if not os.path.exists(shap_path):
        raise FileNotFoundError(f'{shap_path} not found — run interpret_xgboost.py first to compute SHAP')
    shap_df = pd.read_csv(shap_path, index_col=0)
    # index contains feature names, e.g., GENE_mRNA or feature_miRNA etc.
    # filter mRNA features
    mrna_feats = [idx for idx in shap_df.index if str(idx).endswith('_mRNA')]
    mrna_genes = [f[:-5] for f in mrna_feats]
    mrna_shap = shap_df.loc[mrna_feats].copy()
    mrna_shap.index = mrna_genes
    # collapse duplicates if any
    mrna_shap = mrna_shap.groupby(mrna_shap.index).mean()
    mrna_shap = mrna_shap.sort_values(by='mean_abs_shap', ascending=False)
    ranked_path = os.path.join(args.outdir, 'ranked_mrna_shap.csv')
    mrna_shap.to_csv(ranked_path)
    print('Saved ranked mRNA shap:', ranked_path)

    # parse top Ns
    try:
        top_ns = sorted({int(x.strip()) for x in args.top.split(',') if x.strip()})
    except Exception:
        top_ns = [150]
    if len(top_ns) == 0:
        top_ns = [150]

    for n in top_ns:
        top_list = mrna_shap.head(n).index.tolist()
        with open(os.path.join(args.outdir, f'top_{n}_genes.txt'), 'w') as fh:
            fh.write('\n'.join(top_list))
        print(f'Saved top_{n}_genes.txt (n={len(top_list)})')

    # parse CellMarker file into gene sets (cell types)
    if not os.path.exists(args.cellmarker):
        raise FileNotFoundError(f'CellMarker file not found at {args.cellmarker}')
    gs = parse_cellmarker(args.cellmarker)
    print('Parsed gene sets (cell types):', len(gs))

    # run ORA for each top-N
    bg_set = set(background_genes)
    for n in top_ns:
        top_list = mrna_shap.head(n).index.tolist()
        rows = []
        for name, genes in gs.items():
            genes_in_bg = set(genes) & bg_set
            K = len(genes_in_bg)
            if K == 0:
                continue
            overlap = sorted(set(top_list) & genes_in_bg)
            k = len(overlap)
            p = hypergeom_pval(N, K, n, k)
            enr_ratio = (k / n) / (K / N) if (K > 0 and n > 0) else np.nan
            rows.append({
                'gene_set': name,
                'overlap_count': k,
                'overlapping_genes': ';'.join(overlap),
                'query_size_n': n,
                'background_size_N': N,
                'gene_set_size_in_bg': K,
                'raw_p_value': p,
                'enrichment_ratio': enr_ratio,
            })
        if len(rows) == 0:
            print(f'No gene sets with members in background for top {n}, skipping')
            continue
        ora_df = pd.DataFrame(rows).sort_values('raw_p_value')
        out_raw = os.path.join(args.outdir, f'ora_top{n}_raw.csv')
        ora_df.to_csv(out_raw, index=False)
        ora_df['adj_p_value'] = bh_adjust(ora_df['raw_p_value'].values)
        ora_df = ora_df.sort_values('adj_p_value')
        out_adj = os.path.join(args.outdir, f'ora_top{n}_adj.csv')
        ora_df.to_csv(out_adj, index=False)
        sig_df = ora_df[ora_df['adj_p_value'] <= 0.05]
        out_sig = os.path.join(args.outdir, f'ora_top{n}_significant.csv')
        sig_df.to_csv(out_sig, index=False)
        print(f'ORA done for top {n}: {len(ora_df)} sets, {len(sig_df)} significant')

        # plot top 20 by adj p
        plot_df = ora_df.head(20).copy()
        if not plot_df.empty:
            plot_df['minus_log10_adj'] = -np.log10(plot_df['adj_p_value'].replace(0, 1e-300))
            plt.figure(figsize=(8, max(4, 0.25 * len(plot_df))))
            sns.scatterplot(data=plot_df, x='minus_log10_adj', y='gene_set',
                            size='overlap_count', hue='minus_log10_adj', palette='viridis',
                            sizes=(50, 300), legend=False)
            plt.xlabel('-log10(adj_p_value)')
            plt.ylabel('Gene set / cell type')
            plt.title(f'ORA (top {n} mRNA genes) — top enriched cell types')
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f'ora_top{n}_dotplot.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print('Saved dotplot for top', n)

    # summary metadata
    meta = {
        'shap_source': 'interpret_xgboost.py shap_top20.csv',
        'shap_sample_size': None,
        'top_ns': top_ns,
        'background_size': N,
        'gene_set_library': os.path.basename(args.cellmarker),
        'stat_test': 'hypergeometric (exact)',
        'multiple_test_correction': 'Benjamini-Hochberg FDR'
    }
    with open(os.path.join(args.outdir, 'ORA_metadata.json'), 'w') as fh:
        import json
        fh.write(json.dumps(meta, indent=2))
    print('Saved metadata')


if __name__ == '__main__':
    main()
