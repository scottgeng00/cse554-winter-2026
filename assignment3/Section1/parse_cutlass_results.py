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

best = combined.groupby(['m', 'n', 'k'])['GFLOPs'].max().reset_index()
best.columns = ['batch_size', 'N', 'K', 'gflops']
best['tflops'] = best['gflops'] / 1000.0
best['library'] = 'cutlass'

existing = pd.read_csv('Section1/gemm_perf.csv')

cutlass_out = best[['batch_size', 'N', 'K', 'library', 'tflops']]
merged = pd.concat([existing, cutlass_out], ignore_index=True)
merged = merged.sort_values(['N', 'K', 'library', 'batch_size']).reset_index(drop=True)
merged.to_csv('Section1/gemm_perf.csv', index=False)

print(f"Added {len(cutlass_out)} CUTLASS rows to gemm_perf.csv")
print(f"Total rows: {len(merged)}")
print("\nCUTLASS best TFLOPS per shape:")
print(cutlass_out.groupby(['N', 'K'])['tflops'].max())
