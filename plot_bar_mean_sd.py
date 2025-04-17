import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/pelvis_motion_data_fake.csv')

# 제외할 컬럼 지정
EXCLUDE_COLS = ['id', 'label']
metrics = sorted(df.select_dtypes(include='number').columns.difference(EXCLUDE_COLS))

for metric in metrics:
    plt.figure(figsize=(6, 4))

    ax = sns.barplot(
        x='label',
        y=metric,
        data=df,
        errorbar='sd',
        capsize=0.1,
        err_kws={'linewidth': 1}
    )

    # 막대 안쪽 가운데에 평균값 표시
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            f'{height:.5f}',
            ha='center',
            va='center',
            color='white',
            fontsize=10,
            fontweight='bold'
        )

    plt.title(f'{metric} (Mean ± SD)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
