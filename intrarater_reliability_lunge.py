import pandas as pd
import pingouin as pg

# CSV 파일 불러오기
df = pd.read_csv('./data/reliability/data_measurement_filament.csv')

# 열 이름 정리
df.columns = ['Name', 'Trial1', 'Trial2']

# long-format으로 변환
df_long = pd.melt(df, id_vars='Name', value_vars=['Trial1', 'Trial2'], var_name='Rater', value_name='Score')

# ICC 계산
icc = pg.intraclass_corr(data=df_long, targets='Name', raters='Rater', ratings='Score')
icc_filtered = icc[icc['Type'].isin(['ICC3', 'ICC3k'])]

# 결과 출력
print(icc_filtered[['Type', 'ICC', 'CI95%']])
