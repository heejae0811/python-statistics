import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# CSV 파일 불러오기
df = pd.read_csv('./data/reliability/data_measurement_swm.csv')

# 열 이름 정리
df.columns = ['Name', 'Trial1', 'Trial2']

# Pearson 상관분석
r, p = pearsonr(df['Trial1'], df['Trial2'])
print(f"Pearson 상관계수 r = {r:.4f}, p-value = {p:.4f}")

# 산점도 + 회귀선 시각화
plt.figure(figsize=(6, 4))
plt.scatter(df['Trial1'], df['Trial2'], color='steelblue', label='Subjects')

# 회귀선 그리기
m, b = np.polyfit(df['Trial1'], df['Trial2'], 1)
plt.plot(df['Trial1'], m * df['Trial1'] + b, color='darkred', linestyle='--', label='Linear Fit')

# 그래프 설정
plt.xlabel('SWM Trial 1')
plt.ylabel('SWM Trial 1')
plt.title('Correlation Between Trial 1 and Trial 2')
plt.text(min(df['Trial1']), max(df['Trial2']) - 0.5, f'r = {r:.4f}, p = {p:.4f}', fontsize=10)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 시각화 표시
plt.show()