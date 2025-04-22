# 산점도 행렬 (pair plot)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/pelvis_motion_data_fake.csv')

graph = sns.pairplot(
    df,                     # 데이터프레임
    hue="label",            # 라벨 컬럼 (범주에 따라 색상 다르게)
    diag_kind="kde",        # 대각선에는 밀도 그래프 (기본값은 'auto')
    plot_kws={'alpha': 0.5} # 산점도 투명도 조절
)

plt.show()