import pandas as pd
import glob
import os

results_dir = 'Section1/cutlass_results'
csv_files = glob.glob(os.path.join(results_dir, '*.gemm.csv'))

all_rows = []
for f in csv_files:
    df = pd.read_csv(f)
    df = df[df['Status'] == 'success']
    all_rows.append(df)

combined = pd.concat(all_rows, ignore_index=True)

idx = combined.groupby(['m', 'n', 'k'])['GFLOPs'].idxmax()
best = combined.loc[idx]

cols = ['m', 'n', 'k', 'split_k_slices', 'cta_m', 'cta_n', 'cta_k', 'op_class', 'GFLOPs', 'Operation']

for (n, k), group in best.groupby(['n', 'k']):
    print(f"\n{'='*80}")
    print(f"Shape N={n}, K={k}")
    print(f"{'='*80}")
    sub = group.sort_values('m')[cols]
    for _, row in sub.iterrows():
        print(f"  M={int(row['m']):5d}  |  tile=({int(row['cta_m'])}x{int(row['cta_n'])}x{int(row['cta_k'])})  "
              f"split_k={int(row['split_k_slices'])}  {row['op_class']:>8s}  "
              f"{row['GFLOPs']:8.1f} GFLOPs")

print(f"\n\n{'='*80}")
print("Split-K usage summary")
print(f"{'='*80}")
sk_counts = best.groupby(['n', 'k', 'split_k_slices']).size().reset_index(name='count')
for (n, k), grp in sk_counts.groupby(['n', 'k']):
    print(f"\n  N={n}, K={k}:")
    for _, row in grp.iterrows():
        print(f"    split_k={int(row['split_k_slices'])}: won {row['count']} times")

print(f"\n\n{'='*80}")
print("Tile size usage summary")
print(f"{'='*80}")
best['tile'] = best.apply(lambda r: f"{int(r['cta_m'])}x{int(r['cta_n'])}x{int(r['cta_k'])}", axis=1)
tile_counts = best.groupby(['n', 'k', 'tile']).size().reset_index(name='count')
for (n, k), grp in tile_counts.groupby(['n', 'k']):
    print(f"\n  N={n}, K={k}:")
    for _, row in grp.sort_values('count', ascending=False).iterrows():
        print(f"    tile={row['tile']}: won {row['count']} times")
