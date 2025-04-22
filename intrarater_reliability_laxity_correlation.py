import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1. CSV 파일 불러오기
df = pd.read_csv('./data/reliability/data_measurement_laxity.csv', encoding='utf-8-sig')

# 2. 열 이름 정리
df.columns = ['ID', 'L_Anterior_1', 'L_Tilt_1', 'R_Anterior_1', 'R_Tilt_1', 'L_Anterior_2', 'L_Tilt_2', 'R_Anterior_2', 'R_Tilt_2']

# 3. 분석할 항목 리스트
metrics = ['L_Anterior', 'L_Tilt', 'R_Anterior', 'R_Tilt']
results = []

# 4. 상관계수 계산 및 그래프 출력
for metric in metrics:
    x = df[f'{metric}_1']
    y = df[f'{metric}_2']

    r, p = pearsonr(x, y)
    results.append({
        'Metric': metric,
        'Pearson r': round(r, 3),
        'p-value': round(p, 4)
    })

    # 그래프 그리기
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='steelblue', label='Subjects')
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color='darkred', linestyle='--', label='Linear Fit')

    plt.xlabel(f'{metric} - Trial 1')
    plt.ylabel(f'{metric} - Trial 2')
    plt.title(f'Correlation: {metric}')
    plt.text(min(x), max(y) - 0.5, f'r = {r:.3f}, p = {p:.4f}', fontsize=10)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 바로 시각화 출력
    plt.show()

# 5. 결과 출력
correlation_df = pd.DataFrame(results)
print("\n[Correlation Analysis Results]")
print(correlation_df)
