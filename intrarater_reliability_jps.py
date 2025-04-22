import pandas as pd
import pingouin as pg

# 1. 데이터 불러오기
df = pd.read_csv('./data/reliability/data_measurement_jps.csv')

# 2. 열 이름 정리 (필요한 경우만)
df.columns = df.columns.str.strip()  # 공백 제거
df.rename(columns={
    'JPS IV Mean 1': 'IV_1',
    'JPS IV Mean 2': 'IV_2',
    'JPS PF Mean 1': 'PF_1',
    'JPS PF Mean 2': 'PF_2'
}, inplace=True)

# 3. 분석할 항목
metrics = {
    'JPS_IV': ['IV_1', 'IV_2'],
    'JPS_PF': ['PF_1', 'PF_2']
}

# 4. ICC 결과 저장
results = []

for name, (col1, col2) in metrics.items():
    temp = df[['Unnamed: 0', col1, col2]].copy()
    temp_long = pd.melt(temp, id_vars='Unnamed: 0', value_vars=[col1, col2], var_name='Rater', value_name='Score')

    icc = pg.intraclass_corr(data=temp_long, targets='Unnamed: 0', raters='Rater', ratings='Score')

    icc3 = icc[icc['Type'] == 'ICC3'].iloc[0]
    icc3k = icc[icc['Type'] == 'ICC3k'].iloc[0]

    results.append({
        'List': name,
        'ICC(3,1)': round(icc3['ICC'], 3),
        'ICC(3,k)': round(icc3k['ICC'], 3),
        '95% CI(3,1)': icc3['CI95%'],
        '95% CI(3,k)': icc3k['CI95%']
    })

# 5. 결과 출력
icc_df = pd.DataFrame(results)
print(icc_df)
