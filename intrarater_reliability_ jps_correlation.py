import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_csv('./data/reliability/data_measurement_jps.csv')

# 2. 열 이름 정리
df.columns = ['Name',
              'JPS_IV_1', 'JPS_IV_2', 'JPS_IV_3', 'JPS_IV_Mean1',
              'JPS_IV_1_2', 'JPS_IV_2_2', 'JPS_IV_3_2', 'JPS_IV_Mean2',
              'JPS_PF_1', 'JPS_PF_2', 'JPS_PF_3', 'JPS_PF_Mean1',
              'JPS_PF_1_2', 'JPS_PF_2_2', 'JPS_PF_3_2', 'JPS_PF_Mean2']

# 3. 피어슨 상관분석
iv_r, iv_p = pearsonr(df['JPS_IV_Mean1'], df['JPS_IV_Mean2'])
pf_r, pf_p = pearsonr(df['JPS_PF_Mean1'], df['JPS_PF_Mean2'])

# 4. 결과 출력
print("JPS_IV 상관분석 결과")
print(f"상관계수 r = {iv_r:.4f}, p-value = {iv_p:.4f}")

print("\nJPS_PF 상관분석 결과")
print(f"상관계수 r = {pf_r:.4f}, p-value = {pf_p:.4f}")

# 5. JPS_IV 산점도 + 회귀선
plt.figure(figsize=(6, 4))
plt.scatter(df['JPS_IV_Mean1'], df['JPS_IV_Mean2'], color='steelblue', label='Subjects')
m_iv, b_iv = np.polyfit(df['JPS_IV_Mean1'], df['JPS_IV_Mean2'], 1)
plt.plot(df['JPS_IV_Mean1'], m_iv * df['JPS_IV_Mean1'] + b_iv, color='darkred', linestyle='--', label='Linear Fit')
plt.xlabel('JPS IV Trial 1 (Mean)')
plt.ylabel('JPS IV Trial 2 (Mean)')
plt.title('Correlation - Inversion (JPS IV)')
plt.text(min(df['JPS_IV_Mean1']), max(df['JPS_IV_Mean2']) - 0.5, f'r = {iv_r:.4f}, p = {iv_p:.4f}', fontsize=10)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 6. JPS_PF 산점도 + 회귀선
plt.figure(figsize=(6, 4))
plt.scatter(df['JPS_PF_Mean1'], df['JPS_PF_Mean2'], color='mediumseagreen', label='Subjects')
m_pf, b_pf = np.polyfit(df['JPS_PF_Mean1'], df['JPS_PF_Mean2'], 1)
plt.plot(df['JPS_PF_Mean1'], m_pf * df['JPS_PF_Mean1'] + b_pf, color='darkred', linestyle='--', label='Linear Fit')
plt.xlabel('JPS PF Trial 1 (Mean)')
plt.ylabel('JPS PF Trial 2 (Mean)')
plt.title('Correlation - Plantarflexion (JPS PF)')
plt.text(min(df['JPS_PF_Mean1']), max(df['JPS_PF_Mean2']) - 0.5, f'r = {pf_r:.4f}, p = {pf_p:.4f}', fontsize=10)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
