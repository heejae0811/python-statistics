import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# 데이터 불러오기
df = pd.read_csv('./pelvis_motion_data_fake.csv')

# 특성과 레이블 분리
X = df.drop(['id', 'label'], axis=1)
y = df['label']

# 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 하이퍼파라미터 그리드 정의
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced']
}

# 모델 정의
dt_classifier = DecisionTreeClassifier(random_state=42)

# GridSearchCV 정의
grid_search = GridSearchCV(
    estimator=dt_classifier,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

# 학습
grid_search.fit(X_train, y_train)

# 최적 모델 평가
y_pred = grid_search.best_estimator_.predict(X_test)
y_prob = grid_search.best_estimator_.predict_proba(X_test)[:, 1]

print("Best parameters: ", grid_search.best_params_)
print("Best f1 score (CV 평균): ", round(grid_search.best_score_, 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# GridSearchCV 결과 요약
cv_results = pd.DataFrame(grid_search.cv_results_)
result_df = cv_results[[
    'mean_test_score', 'std_test_score', 'rank_test_score',
    'param_criterion', 'param_max_depth',
    'param_min_samples_split', 'param_min_samples_leaf',
    'param_class_weight'
]].sort_values(by='mean_test_score', ascending=False)
print("\nTop 10 parameter combinations by mean_test_score:")
print(result_df.head(10).to_string(index=False))

# Confusion Matrix 시각화
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Rehab", "Return"])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# 결정 트리 시각화
best_tree = grid_search.best_estimator_
plt.figure(figsize=(30, 10))
plot_tree(best_tree,
          feature_names=X.columns,
          class_names=['Rehab', 'Return'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Best Decision Tree')
plt.show()
print("Max Depth of the best tree:", best_tree.get_depth())

# Feature Importance 시각화
feature_importances = best_tree.feature_importances_
indices = np.argsort(feature_importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(X.shape[1]), feature_importances[indices], align="center")
plt.yticks(range(X.shape[1]), features[indices])
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# ROC Curve 시각화
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# TPR, FPR 직접 계산한 값 예시 (수동 계산용)
TP = 58
FP = 5
FN = 5
TN = 102

TPR_recall = TP / (TP + FN)
FPR_rate = FP / (FP + TN)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, color='blue', lw=1, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.scatter(FPR_rate, TPR_recall, color='red', label=f'TPR={TPR_recall:.2f}, FPR={FPR_rate:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Manual TPR/FPR')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
