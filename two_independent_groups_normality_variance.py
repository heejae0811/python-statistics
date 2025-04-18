import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

df = pd.read_csv('data/pelvis_motion_data_random.csv')

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
                t_stat, t_p = ttest_ind(group_data[groups[0]], group_data[groups[1]], equal_var=True)
                print(f'\n→ Independent t-test | stat: {t_stat:.5f} | p-value: {t_p:.5f}')
            else:
                print(f'→ Levene’s test | stat: {levene_stat:.5f} | p-value: {levene_p:.5f} | 등분산이 아니다.')

                # Welch's t-test
                t_stat, t_p = ttest_ind(group_data[groups[0]], group_data[groups[1]], equal_var=False)
                print(f'\n→ Welch’s t-test | stat: {t_stat:.5f} | p-value: {t_p:.5f}')
        else:
            print('정규분포가 아니기 때문에 등분산검정 생략')

            # Mann–Whitney U test
            u_stat, u_p = mannwhitneyu(group_data[groups[0]], group_data[groups[1]], alternative='two-sided')
            print(f'\n→ Mann–Whitney U test | stat: {u_stat:.5f} | p-value: {u_p:.5f}')
    else:
        print('샘플 수 부족')