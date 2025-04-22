import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1. 데이터 불러오기
df = pd.read_csv('./data/reliability/data_measurement_jps.csv')

# 2. 열 이름 정리
df.rename(columns={
    'JPS IV Mean 1': 'IV_1',
    'JPS IV Mean 2': 'IV_2',
    'JPS PF Mean 1': 'PF_1',
    'JPS PF Mean 2': 'PF_2'
}, inplace=True)

# 3. 분석할 항목 리스트
metrics = [('IV_1', 'IV_2', 'JPS - IV'), ('PF_1', 'PF_2', 'JPS - PF')]
results = []

# 4. 각 항목에 대해 Pearson 상관계수 계산 및 시각화
for col1, col2, label in metrics:
    x = df[col1]
    y = df[col2]

    r, p = pearsonr(x, y)
    results.append({
        '항목': label,
        'Pearson r': round(r, 3),
        'p-value': round(p, 4)
    })

    # 그래프 그리기
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='steelblue', label='Subjects')
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color='darkred', linestyle='--', label='Linear Fit')

    plt.xlabel(f'{label} - Trial 1')
    plt.ylabel(f'{label} - Trial 2')
    plt.title(f'Correlation: {label}')
    plt.text(min(x), max(y) - 0.5, f'r = {r:.3f}, p = {p:.4f}', fontsize=10)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 5. 결과 출력
cor_df = pd.DataFrame(results)
print("\n[상관관계 분석 결과]")
print(cor_df)
