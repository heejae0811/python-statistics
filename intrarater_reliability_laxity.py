import pandas as pd
import pingouin as pg

# 1. CSV 파일 불러오기
df = pd.read_csv('./data/reliability/data_measurement_laxity.csv')

# 2. 열 이름 정리
df.columns = ['ID', 'L_Anterior_1', 'L_Tilt_1', 'R_Anterior_1', 'R_Tilt_1', 'L_Anterior_2', 'L_Tilt_2', 'R_Anterior_2', 'R_Tilt_2']

# 3. 분석 대상 항목
metrics = ['L_Anterior', 'L_Tilt', 'R_Anterior', 'R_Tilt']
results = []

# 4. 각 항목별 ICC(3,1), ICC(3,k) 계산
for metric in metrics:
    df_temp = df[['ID', f'{metric}_1', f'{metric}_2']].copy()
    df_long = pd.melt(df_temp, id_vars='ID', value_vars=[f'{metric}_1', f'{metric}_2'], var_name='Rater', value_name='Score')

    icc = pg.intraclass_corr(data=df_long, targets='ID', raters='Rater', ratings='Score')

    icc3 = icc[icc['Type'] == 'ICC3'].iloc[0]
    icc3k = icc[icc['Type'] == 'ICC3k'].iloc[0]

    results.append({
        '측정 항목': metric,
        'ICC(3,1)': round(icc3['ICC'], 3),
        'ICC(3,k)': round(icc3k['ICC'], 3),
        '95% CI (3,1)': icc3['CI95%'],
        '95% CI (3,k)': icc3k['CI95%']
    })

# 5. 결과 출력
icc_df = pd.DataFrame(results)
print(icc_df)
