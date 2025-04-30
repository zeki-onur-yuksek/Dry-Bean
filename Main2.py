import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)

# -------------------------------
# Bölüm 1: Veri Ön İşleme
# -------------------------------

# 1. Veri setini oku
df = pd.read_excel("Dry_Bean_Dataset.xlsx")

# 2. Eksik veri ekle
#    Rastgele %5 eksik: örnek olarak 'Area' ve 'Perimeter'
for col in ['Area', 'Perimeter']:
    df.loc[df.sample(frac=0.05, random_state=1).index, col] = np.nan
#    Rastgele %35 eksik: 'Eccentricity'
df.loc[df.sample(frac=0.35, random_state=2).index, 'Eccentricity'] = np.nan

# a) Eksik değerleri gözlemle
print("=== Eksik Değer Sayısı ===")
print(df.isnull().sum(), "\n")

# b) %5’lik eksikleri median ile doldur
for col in ['Area', 'Perimeter']:
    df[col].fillna(df[col].median(), inplace=True)

# c) %35’lik eksik sütunu kaldır
df.drop(columns=['Eccentricity'], inplace=True)

# 3. Aykırı değer tespiti ve sınırlandırma (Z-score yöntemiyle)
#    Aykırıları drop yerine cap yöntemiyle [mean±3*std] aralığına getiriyoruz.
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('Class_Label', errors='ignore')
for col in numeric_cols:
    mean_, std_ = df[col].mean(), df[col].std()
    lower, upper = mean_ - 3*std_, mean_ + 3*std_
    df[col] = np.clip(df[col], lower, upper)

# 4. Özellik ölçekleme
scaler = StandardScaler()

# 5. Kategorik kodlama: sınıf etiketlerini numerik yap
label_encoder = LabelEncoder()
df['Class_Label'] = label_encoder.fit_transform(df['Class'])

# 6. Modelleme için girdi matrisi ve hedef
X_full = df.drop(columns=['Class', 'Class_Label'])
X_scaled = pd.DataFrame(scaler.fit_transform(X_full), columns=X_full.columns)
y = df['Class_Label'].values

# -------------------------------
# Bölüm 2: Özellik Seçimi ve Boyut İndirgeme
# -------------------------------

# 7. PCA ile boyut indirgeme
pca_temp = PCA().fit(X_scaled)
explained = pca_temp.explained_variance_ratio_
n_pca = max(2, np.sum(explained > explained.mean()))  # en az 2 bileşen
pca = PCA(n_components=n_pca)
X_pca = pca.fit_transform(X_scaled)

# PCA’nın ilk iki bileşeni ile scatter
plt.figure(figsize=(7,5))
for cls in np.unique(y):
    idx = y == cls
    plt.scatter(X_pca[idx,0], X_pca[idx,1], s=30, label=str(cls))
plt.title("PCA: İlk 2 Bileşen ile Sınıf Ayrımı")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(title="Class", bbox_to_anchor=(1,1))
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. LDA ile boyut indirgeme
n_classes = len(np.unique(y))
n_lda = min(n_classes - 1, 3)
lda = LDA(n_components=n_lda)
X_lda = lda.fit_transform(X_scaled, y)

# LDA’nın ilk iki bileşeni ile scatter
plt.figure(figsize=(7,5))
for cls in np.unique(y):
    idx = y == cls
    plt.scatter(X_lda[idx,0], X_lda[idx,1], s=30, label=str(cls))
plt.title("LDA: İlk 2 Bileşen ile Sınıf Ayrımı")
plt.xlabel("LD1"); plt.ylabel("LD2")
plt.legend(title="Class", bbox_to_anchor=(1,1))
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Bölüm 3: Modelleme ve Değerlendirme
# -------------------------------

# 9. Nested CV + modeller + temsil biçimleri
representations = {
    'raw': X_scaled.values,
    'pca': X_pca,
    'lda': X_lda
}

models = {
    'LogisticRegression': (
        LogisticRegression(max_iter=1000, random_state=42),
        {'C':[0.01,0.1,1,10]}
    ),
    'DecisionTree': (
        DecisionTreeClassifier(random_state=42),
        {'max_depth':[None,5,10]}
    ),
    'RandomForest': (
        RandomForestClassifier(random_state=42),
        {'n_estimators':[100,200], 'max_depth':[None,10]}
    ),
    'XGBoost': (
        XGBClassifier(eval_metric='mlogloss', random_state=42),
        {'n_estimators':[100,200], 'max_depth':[3,6]}
    ),
    'NaiveBayes': (
        GaussianNB(),
        {'var_smoothing':[1e-9,1e-8,1e-7]}
    )
}

# Sonuçları biriktireceğimiz yapı
results = {
    rep: {m:{'acc':[],'prec':[],'rec':[],'f1':[]} for m in models}
    for rep in representations
}
roc_data = {}

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for rep_name, X in representations.items():
    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for name, (clf, params) in models.items():
            gs = GridSearchCV(clf, params, cv=inner_cv, scoring='f1_macro')
            gs.fit(X_tr, y_tr)
            best = gs.best_estimator_

            y_pred = best.predict(X_te)
            y_proba = best.predict_proba(X_te)

            # metrikleri kaydet
            results[rep_name][name]['acc'].append(accuracy_score(y_te,y_pred))
            results[rep_name][name]['prec'].append(precision_score(y_te,y_pred,average='macro'))
            results[rep_name][name]['rec'].append(recall_score(y_te,y_pred,average='macro'))
            results[rep_name][name]['f1'].append(f1_score(y_te,y_pred,average='macro'))

            # en iyi F1 için ROC verisi sakla
            key = (rep_name,name)
            f1score = f1_score(y_te,y_pred,average='macro')
            if key not in roc_data or roc_data[key]['f1'] < f1score:
                roc_data[key] = {'y_te':y_te, 'y_score':y_proba, 'f1':f1score}

# 10. Ortalama std ile raporla
for rep in results:
    print(f"\n--- Representation: {rep} ---")
    for m in results[rep]:
        arr = results[rep][m]
        print(f"{m:15s} | "
              f"Acc: {np.mean(arr['acc']):.3f}±{np.std(arr['acc']):.3f} | "
              f"Prec: {np.mean(arr['prec']):.3f}±{np.std(arr['prec']):.3f} | "
              f"Rec: {np.mean(arr['rec']):.3f}±{np.std(arr['rec']):.3f} | "
              f"F1: {np.mean(arr['f1']):.3f}±{np.std(arr['f1']):.3f}")

# 11. En iyi model için ROC eğrileri (OVA)
best_key = max(roc_data, key=lambda k: roc_data[k]['f1'])
y_te_best = roc_data[best_key]['y_te']
y_score_best = roc_data[best_key]['y_score']
n_cls = len(np.unique(y))

y_bin = label_binarize(y_te_best, classes=range(n_cls))
plt.figure(figsize=(7,6))
for i in range(n_cls):
    fpr, tpr, _ = roc_curve(y_bin[:,i], y_score_best[:,i])
    plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC={auc(fpr,tpr):.2f})")

plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title(f"ROC Curves for Best Model: {best_key[0]} / {best_key[1]}")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
