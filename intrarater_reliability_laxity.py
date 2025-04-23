import pandas as pd
import pingouin as pg

# 1. CSV 파일 불러오기
df = pd.read_csv('./data/reliability/data_measurement_laxity.csv')

# 2. 열 이름 정리
df.columns = ['Name',
              'Laxity Anterior L 1', 'Laxity Anterior L 2',
              'Laxity Anterior R 1', 'Laxity Anterior R 2',
              'Laxity Talar tilt L 1', 'Laxity Talar tilt L 2',
              'Laxity Talar tilt R 1', 'Laxity Talar tilt R 2']

# 3. 분석할 변수 목록 정의
variables = {
    'Laxity Anterior L': ['Laxity Anterior L 1', 'Laxity Anterior L 2'],
    'Laxity Anterior R': ['Laxity Anterior R 1', 'Laxity Anterior R 2'],
    'Laxity Talar tilt L': ['Laxity Talar tilt L 1', 'Laxity Talar tilt L 2'],
    'Laxity Talar tilt R': ['Laxity Talar tilt R 1', 'Laxity Talar tilt R 2'],
}

# 4. 각 변수에 대해 ICC(3,1) 계산 및 출력
for label, cols in variables.items():
    df_long = pd.DataFrame({
        'Name': df['Name'].tolist() * 2,
        'Rater': ['Trial 1'] * len(df) + ['Trial 2'] * len(df),
        'Score': df[cols[0]].tolist() + df[cols[1]].tolist()
    })

    icc_result = pg.intraclass_corr(data=df_long, targets='Name', raters='Rater', ratings='Score')
    icc_3_1 = icc_result[icc_result['Type'] == 'ICC3']

    print(f"\nICC(3,1) 결과 - {label}")
    print(icc_3_1[['Type', 'ICC', 'CI95%']])
