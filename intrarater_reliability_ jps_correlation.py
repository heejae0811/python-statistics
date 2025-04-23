import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 1. CSV 파일 불러오기
df = pd.read_csv('./data/reliability/data_measurement_jps.csv')
# 2. 평균값 계산
df['JPS_IV_1st'] = df[['JPS IV - 1', 'JPS IV - 2', 'JPS IV - 3']].mean(axis=1)
df['JPS_IV_2nd'] = df[['JPS IV - 1.1', 'JPS IV - 2.1', 'JPS IV - 3.1']].mean(axis=1)

df['JPS_PF_1st'] = df[['JPS PF - 1', 'JPS PF - 2', 'JPS PF - 3']].mean(axis=1)
df['JPS_PF_2nd'] = df[['JPS PF - 1.1', 'JPS PF - 2.1', 'JPS PF - 3.1']].mean(axis=1)

# 3. 피어슨 상관분석
iv_r, iv_p = pearsonr(df['JPS_IV_1st'], df['JPS_IV_2nd'])
pf_r, pf_p = pearsonr(df['JPS_PF_1st'], df['JPS_PF_2nd'])

# 4. JPS IV 그래프
plt.figure(figsize=(6, 4))
plt.scatter(df['JPS_IV_1st'], df['JPS_IV_2nd'], color='steelblue', label='Subjects')
m1, b1 = np.polyfit(df['JPS_IV_1st'], df['JPS_IV_2nd'], 1)
plt.plot(df['JPS_IV_1st'], m1 * df['JPS_IV_1st'] + b1, color='darkred', linestyle='--', label='Linear Fit')
plt.xlabel('JPS IV Trial 1 (Mean)')
plt.ylabel('JPS IV Trial 2 (Mean)')
plt.title('Correlation - Inversion (JPS IV)')
plt.text(min(df['JPS_IV_1st']), max(df['JPS_IV_2nd']) - 0.5, f'r = {iv_r:.4f}, p = {iv_p:.4f}', fontsize=10)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 5. JPS PF 그래프
plt.figure(figsize=(6, 4))
plt.scatter(df['JPS_PF_1st'], df['JPS_PF_2nd'], color='mediumseagreen', label='Subjects')
m2, b2 = np.polyfit(df['JPS_PF_1st'], df['JPS_PF_2nd'], 1)
plt.plot(df['JPS_PF_1st'], m2 * df['JPS_PF_1st'] + b2, color='darkred', linestyle='--', label='Linear Fit')
plt.xlabel('JPS PF Trial 1 (Mean)')
plt.ylabel('JPS PF Trial 2 (Mean)')
plt.title('Correlation - Plantar Flexion (JPS PF)')
plt.text(min(df['JPS_PF_1st']), max(df['JPS_PF_2nd']) - 0.5, f'r = {pf_r:.4f}, p = {pf_p:.4f}', fontsize=10)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
