import pandas as pd
import pingouin as pg

# 1. CSV 불러오기
df = pd.read_csv('./data/reliability/data_measurement_jps.csv')

# 2. JPS IV: 1차/2차 평균값 만들기
df['JPS_IV_1st'] = df[['JPS IV - 1', 'JPS IV - 2', 'JPS IV - 3']].mean(axis=1)
df['JPS_IV_2nd'] = df[['JPS IV - 1.1', 'JPS IV - 2.1', 'JPS IV - 3.1']].mean(axis=1)

# 3. JPS PF: 1차/2차 평균값 만들기
df['JPS_PF_1st'] = df[['JPS PF - 1', 'JPS PF - 2', 'JPS PF - 3']].mean(axis=1)
df['JPS_PF_2nd'] = df[['JPS PF - 1.1', 'JPS PF - 2.1', 'JPS PF - 3.1']].mean(axis=1)

# 4. Long-format 변환 함수
def make_long_format(data, col1, col2, name='Name'):
    df_long = pd.DataFrame({
        name: data['Unnamed: 0'].tolist() * 2,
        'Rater': [col1] * len(data) + [col2] * len(data),
        'Score': data[col1].tolist() + data[col2].tolist()
    })
    return df_long

# 5. ICC 계산 함수
def calculate_icc(df_long, label):
    icc = pg.intraclass_corr(data=df_long, targets='Name', raters='Rater', ratings='Score')
    result = icc[icc['Type'].isin(['ICC3', 'ICC3k'])]
    print(f"\nICC 결과 - {label}")
    print(result[['Type', 'ICC', 'CI95%']])
    return result

# 6. ICC 계산 실행
df_iv_long = make_long_format(df, 'JPS_IV_1st', 'JPS_IV_2nd')
df_pf_long = make_long_format(df, 'JPS_PF_1st', 'JPS_PF_2nd')

icc_iv = calculate_icc(df_iv_long, 'JPS IV')
icc_pf = calculate_icc(df_pf_long, 'JPS PF')
