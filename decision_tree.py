import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay

df = pd.read_csv('./pelvis_motion_data_fake.csv')
X = df.drop(['id', 'label'], axis=1) # 독립변수
y = df['label'] # 종속변수

# 레이블 인코딩 (문자형일 경우 숫자로 변환)
le = LabelEncoder()
y = le.fit_transform(y)

# 1. 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. 기본 결정 트리 모델 정의
param_grid = {
    'criterion': ['gini', 'entropy'],   # 분할 기준
    'max_depth': [3, 5, 10, None],      # 트리 깊이 제한
    'min_samples_split': [2, 5, 10],    # 내부 노드 분할 최소 샘플 수
    'min_samples_leaf': [1, 2, 4],      # 리프 노드 최소 샘플 수
    'class_weight': [None, 'balanced']  # 클래스 불균형 대응
}

# 3. 기본 모델 정의
dt = DecisionTreeClassifier(random_state=42)

# 4. GridSearchCV 정의
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,           # 5-fold 교차 검증
    scoring='f1',   # 성능 기준 (예: f1-score)
    n_jobs=-1,      # CPU 모두 사용
    verbose=1       # 학습 상태 출력
)

# 5. 학습
grid_search.fit(X_train, y_train)

# 6. 최적 모델로 예측
y_pred = grid_search.best_estimator_.predict(X_test)                # 클래스 예측
y_prob = grid_search.best_estimator_.predict_proba(X_test)[:, 1]    # 확률 예측

# 7. 성능 출력
print('Best Parameters:', grid_search.best_params_)
print('Best f1-score (CV 평균): ', round(grid_search.best_score_, 5))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred, digits=5))

# 8. GridSearchCV 결과 요약
cv_results = pd.DataFrame(grid_search.cv_results_)

summary_df = cv_results[[
    'mean_test_score', 'std_test_score', 'rank_test_score',
    'param_criterion', 'param_max_depth',
    'param_min_samples_split', 'param_min_samples_leaf',
    'param_class_weight'
]].sort_values(by='mean_test_score', ascending=False)

print('Top 10 parameter combinations by mean_test_score:')
print(summary_df.head(10).to_string(index=False))

# 그래프 시각화 1. Decision tree 결정 트리
plt.figure(figsize=(4, 4))
plot_tree(
    grid_search.best_estimator_,    # 최적 모델
    feature_names=X.columns,        # 특성 이름
    class_names=['0', '1'],         # 클래스 이름
    filled=True,                    # 색 채우기
    rounded=True                    # 모서리 둥글게
)
plt.title('Best Decision Tree')
plt.tight_layout()
plt.show()

# 그래프 시각화 2. Feature importance 특성 중요도
importances = grid_search.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]  # 중요도 높은 순서로 정렬

plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df.head(10))

# 그래프 시각화 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # y_prob는 predict_proba()[:, 1]
roc_auc = auc(fpr, tpr)  # 면적 계산

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 랜덤선 기준선
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 그래프 시각화 4. Confusion Matrix 혼동 행렬
ConfusionMatrixDisplay.from_estimator(
    grid_search.best_estimator_,
    X_test,
    y_test,
    display_labels=['Class 0', 'Class 1'],
    cmap='Blues',
    values_format='d'
)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
