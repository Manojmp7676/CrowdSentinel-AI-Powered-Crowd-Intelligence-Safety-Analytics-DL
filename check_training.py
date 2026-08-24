import pandas as pd
import json

df = pd.read_csv('runs/detect/models/yolo_ucf_cc50/train/results.csv')

# Get column names
cols = df.columns.tolist()
map50_col = [c for c in cols if 'mAP50' in c and '95' not in c][0]
prec_col = [c for c in cols if 'precision' in c][0]
rec_col = [c for c in cols if 'recall' in c][0]
box_col = [c for c in cols if 'box_loss' in c][0]
cls_col = [c for c in cols if 'cls_loss' in c][0]
dfl_col = [c for c in cols if 'dfl_loss' in c][0]

print('='*60)
print('YOLOv8 Training Results Summary')
print('='*60)
print(f'Total Epochs Trained: {len(df)}')
print(f'Best mAP50: {df[map50_col].max():.4f} (Epoch {df[map50_col].idxmax()+1})')
print(f'Best Precision: {df[prec_col].max():.4f}')
print(f'Best Recall: {df[rec_col].max():.4f}')
print(f'Final Box Loss: {df[box_col].iloc[-1]:.4f}')
print(f'Final Cls Loss: {df[cls_col].iloc[-1]:.4f}')
print(f'Final DFL Loss: {df[dfl_col].iloc[-1]:.4f}')
print('='*60)
print()
print('Last 5 Epochs:')
print(df[['epoch', box_col, cls_col, dfl_col, prec_col, rec_col, map50_col]].tail().to_string(index=False))

# Save full results
results = {
    'total_epochs': len(df),
    'best_map50': float(df[map50_col].max()),
    'best_precision': float(df[prec_col].max()),
    'best_recall': float(df[rec_col].max()),
    'final_box_loss': float(df[box_col].iloc[-1]),
    'final_cls_loss': float(df[cls_col].iloc[-1]),
    'final_dfl_loss': float(df[dfl_col].iloc[-1]),
}

with open('outputs/yolo_training_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nResults saved to outputs/yolo_training_summary.json')
