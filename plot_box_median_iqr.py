import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./pelvis_motion_data_fake.csv')

# 제외할 컬럼 지정
EXCLUDE_COLS = ['id', 'label']
metrics = sorted(df.select_dtypes(include='number').columns.difference(EXCLUDE_COLS))

for metric in metrics:
    plt.figure(figsize=(6, 4))

    ax = sns.boxplot(
        x='label',
        y=metric,
        data=df,
        showfliers=True,
        linewidth=2,
        width=0.4
    )

    group_labels = df['label'].unique()
    for i, group in enumerate(group_labels):
        median_val = df[df['label'] == group][metric].median()
        ax.text(
            i,
            median_val,
            f'{median_val:.2f}',
            ha='center',
            va='center',
            color='white',
            fontsize=10,
            fontweight='semibold'
        )

    plt.title(f'{metric} (Median + IQR)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
