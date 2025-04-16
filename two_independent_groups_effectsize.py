import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

# Effect size
def compute_cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std

def compute_mannwhitney_r(x, y):
    u_stat, _ = mannwhitneyu(x, y, alternative='two-sided')
    n1, n2 = len(x), len(y)

    # 정규 근사에 기반한 Z 계산
    mean_U = n1 * n2 / 2
    std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u_stat - mean_U) / std_U

    # r 계산 (Rosenthal's r)
    r = abs(z) / np.sqrt(n1 + n2)
    return r

df = pd.read_csv('./pelvis_motion_data.csv')

group_col = 'label' # 분류할 그룹
metrics = df.columns.drop(['id', group_col]) # 분석할 변수 외 나머지 제거
groups = df[group_col].dropna().unique() # 중복과 NaN 값 제거 [0, 1]

for metric in metrics:
    print(f'\n[Feature: {metric}]')

    group_data = {
        group: df[df[group_col] == group][metric].dropna()
        for group in groups
    }

    if all(len(data) > 2 for data in group_data.values()):
        all_normal = True # 정규성 통과 여부 저장

        # 정규성 검사
        for group in groups:
            shapiro_stat, shapiro_p = shapiro(group_data[group])

            if shapiro_p >= 0.05:
                print(f'[Group {group}] Shapiro Test | stat: {shapiro_stat:.5f} | p-value: {shapiro_p:.5f} | 정규분포하다.')
            else:
                print(f'[Group {group}] Shapiro Test | stat: {shapiro_stat:.5f} | p-value: {shapiro_p:.5f} | 정규분포 하지 않는다.')
                all_normal = False

        # 등분산 검사
        if all_normal:
            levene_stat, levene_p = levene(group_data[groups[0]], group_data[groups[1]])

            if levene_p >= 0.05:
                print(f'Levene’s test | stat: {levene_stat:.5f} | p-value: {levene_p:.5f} | 등분산이다.')

                # Independent t-test
                t_stat, t_p = ttest_ind(group_data[groups[0]], group_data[groups[1]], equal_var = True)
                d = compute_cohens_d(group_data[groups[0]], group_data[groups[1]])

                print(f'\n→ Independent t-test | stat: {t_stat:.5f} | p-value: {t_p:.5f} | Cohen\'s d: {d:.5f}')
            else:
                print(f'Levene’s test | stat: {levene_stat:.5f} | p-value: {levene_p:.5f} | 등분산이 아니다.')

                # Welch's t-test
                t_stat, t_p = ttest_ind(group_data[groups[0]], group_data[groups[1]], equal_var = False)
                d = compute_cohens_d(group_data[groups[0]], group_data[groups[1]])

                print(f'\n→ Welch’s t-test | stat: {t_stat:.5f} | p-value: {t_p:.5f} | Cohen\'s d: {d:.5f}')
        else:
            print('정규분포가 아니기 때문에 등분산검정 생략')

            # Mann–Whitney U test
            u_stat, u_p = mannwhitneyu(group_data[groups[0]], group_data[groups[1]], alternative = 'two-sided')
            r = compute_mannwhitney_r(group_data[groups[0]], group_data[groups[1]])

            print(f'\n→ Mann–Whitney U test | stat: {u_stat:.5f} | p-value: {u_p:.5f} | r: {r:.5f}')
    else:
        print('샘플 수 부족')