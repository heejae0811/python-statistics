import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

df = pd.read_csv('./pelvis_motion_data.csv')

group_col = 'label' # 분류할 그룹
metrics = df.columns.drop(['id', group_col])  # Group 열 이름 제외한 나머지 저장
results = []

groups = df[group_col].dropna().unique()
group1, group2 = groups[0], groups[1]

for metric in metrics:
    group1_data = df[df[group_col] == group1][metric].dropna()
    group2_data = df[df[group_col] == group2][metric].dropna()

    if len(group1_data) > 2 and len(group2_data) > 2:
        shapiro_1 = shapiro(group1_data)
        shapiro_2 = shapiro(group2_data)

        # 정규분포 검사
        if shapiro_1.pvalue >= 0.05:
            print(f"첫 번째 그룹의 {metric} P-value는 {shapiro_1.pvalue:.2f}로 정규분포한다.")
        else:
            print(f"첫 번째 그룹의 {metric} P-value는 {shapiro_1.pvalue:.2f}로 정규분포하지 않는다.")

        if shapiro_2.pvalue >= 0.05:
            print(f"두 번째 그룹의 {metric} P-value는 {shapiro_2.pvalue:.2f}로 정규분포한다.")
        else:
            print(f"두 번째 그룹의 {metric} P-value는 {shapiro_2.pvalue:.2f}로 정규분포하지 않는다.")

        # 등분산 검사
        levene_result = levene(group1_data, group2_data)

        if levene_result.pvalue >= 0.05:
            print(f"두 그룹의 {metric} 등분산 검정 결과는 {levene_result.pvalue:.2f}로 등분산하다.\n")
        else:
            print(f"두 그룹의 {metric} 등분산 검정 결과는 {levene_result.pvalue:.2f}로 등분산하지 않는다.\n")

        # 평균, 표준편차
        group1_mean = group1_data.mean()
        group1_std = group1_data.std()
        group2_mean = group2_data.mean()
        group2_std = group2_data.std()

        # 정규분포하다.
        if shapiro_1.pvalue > 0.05 and shapiro_2.pvalue > 0.05:
            # 등분산하다.
            if levene_result.pvalue > 0.05:
                test_name = 'Independent T-test'
                test_result = ttest_ind(group1_data, group2_data, equal_var=True)
            # 등분산하지 않는다.
            else:
                test_name = 'Welch\'s T-test'
                test_result = ttest_ind(group1_data, group2_data, equal_var=False)
        # 정규분포하지 않는다.
        else:
            test_name = 'Mann-Whitney U test'
            test_result = mannwhitneyu(group1_data, group2_data, alternative='two-sided')

        statistic = test_result.statistic
        p_value = test_result.pvalue

        results.append({
            'Test Used': test_name,
            'Variables': metric,
            f'{group1} Group (mean ± std)': f"{group1_mean:.5f} ± {group1_std:.2f}",
            f'{group2} Group (mean ± std)': f"{group2_mean:.5f} ± {group2_std:.2f}",
            'Test Statistic': f"{statistic:.5f}",
            'P-value': f"{p_value:.5f}"
        })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))