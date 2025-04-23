import pandas as pd
import pingouin as pg

# 1. CSV 파일 불러오기
df = pd.read_csv('./data/reliability/data_measurement_jps.csv')

# 2. 열 이름 정리
df.columns = ['Name',
              'JPS_IV_1', 'JPS_IV_2', 'JPS_IV_3', 'JPS_IV_Mean1',
              'JPS_IV_1_2', 'JPS_IV_2_2', 'JPS_IV_3_2', 'JPS_IV_Mean2',
              'JPS_PF_1', 'JPS_PF_2', 'JPS_PF_3', 'JPS_PF_Mean1',
              'JPS_PF_1_2', 'JPS_PF_2_2', 'JPS_PF_3_2', 'JPS_PF_Mean2']

# 3. JPS_IV 평균값 long-format 변환
df_iv_long = pd.DataFrame({
    'Name': df['Name'].tolist() * 2,
    'Rater': ['Mean1'] * len(df) + ['Mean2'] * len(df),
    'Score': df['JPS_IV_Mean1'].tolist() + df['JPS_IV_Mean2'].tolist()
})

# 4. JPS_PF 평균값 long-format 변환
df_pf_long = pd.DataFrame({
    'Name': df['Name'].tolist() * 2,
    'Rater': ['Mean1'] * len(df) + ['Mean2'] * len(df),
    'Score': df['JPS_PF_Mean1'].tolist() + df['JPS_PF_Mean2'].tolist()
})

# 5. ICC 계산 함수
def calculate_icc(data, label):
    icc_result = pg.intraclass_corr(data=data, targets='Name', raters='Rater', ratings='Score')
    icc_k = icc_result[icc_result['Type'] == 'ICC3k']
    print(f"\nICC(3,k) 결과 - {label}")
    print(icc_k[['Type', 'ICC', 'CI95%']])
    return icc_k

# 6. ICC(3,k) 결과 출력
icc_iv = calculate_icc(df_iv_long, "JPS_IV")
icc_pf = calculate_icc(df_pf_long, "JPS_PF")
