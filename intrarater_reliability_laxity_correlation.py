import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

# 1. CSV 파일 불러오기
df = pd.read_csv('./data/reliability/data_measurement_laxity.csv')

# 2. 열 이름 정리
df.columns = ['Name',
              'Anterior Drawer Test L 1', 'Anterior Drawer Test L 2',
              'Anterior Drawer Test R 1', 'Anterior Drawer Test R 2',
              'Talar Tilt Inversion L 1', 'Talar Tilt Inversion L 2',
              'Talar Tilt Inversion R 1', 'Talar Tilt Inversion R 2']

# 3. 분석할 변수 목록 정의
variables = {
    'Anterior Drawer Test L': ['Anterior Drawer Test L 1', 'Anterior Drawer Test L 2'],
    'Anterior Drawer Test R': ['Anterior Drawer Test R 1', 'Anterior Drawer Test R 2'],
    'Talar Tilt Inversion L': ['Talar Tilt Inversion L 1', 'Talar Tilt Inversion L 2'],
    'Talar Tilt Inversion R': ['Talar Tilt Inversion R 1', 'Talar Tilt Inversion R 2'],
}

# 4. 상관분석 및 그래프 그리기
for label, cols in variables.items():
    x = df[cols[0]]
    y = df[cols[1]]

    # 피어슨 상관분석
    r, p = pearsonr(x, y)
    print(f"\n{label}")
    print(f"상관계수 r = {r:.4f}, p-value = {p:.4f}")

    # 산점도 + 회귀선 시각화
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color='teal', label='Subjects')

    # 회귀선
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color='darkred', linestyle='--', label='Linear Fit')

    # 라벨 및 텍스트
    plt.xlabel(f'{label} - Trial 1')
    plt.ylabel(f'{label} - Trial 2')
    plt.title(f'Correlation - {label}')
    plt.text(min(x), max(y), f'r = {r:.4f}, p = {p:.4f}', fontsize=10)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
