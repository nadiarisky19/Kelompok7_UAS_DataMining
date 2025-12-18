# =====================================================
#  IMPORT LIBRARY
# =====================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pyswarms as ps

# =====================================================
#  HELPER METRICS
# =====================================================
def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn + 1e-12)

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-12)

# =====================================================
#  SUPERVISED FITNESS (PSO)
# =====================================================
def supervised_fitness(centers, X, y, m):
    diff = X[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12

    expo = 2.0 / (m - 1.0)
    U = 1.0 / np.sum((dist[:, :, None] / dist[:, None, :]) ** expo, axis=2)

    labels = np.argmax(U, axis=1)
    acc1 = accuracy_score(y, labels)
    acc2 = accuracy_score(y, 1 - labels)

    return 1 - max(acc1, acc2)

# =====================================================
#  LOAD DATA
# =====================================================
df_path = r"E:\Dokument\Semester 7\Data Mining\Kelompok 7_UAS\diabetes.csv"
df = pd.read_csv(df_path)

# =====================================================
#  PREPROCESSING
# =====================================================
df = df.drop_duplicates().reset_index(drop=True)

# Hapus nilai 0 tidak valid
df_clean = df[(df["Glucose"] != 0) &
              (df["BloodPressure"] != 0) &
              (df["BMI"] != 0)].copy()

# Outlier handling (IQR)
def remove_outliers_iqr(df, cols):
    df_out = df.copy()
    for col in cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        df_out = df_out[(df_out[col] >= Q1 - 1.5*IQR) &
                        (df_out[col] <= Q3 + 1.5*IQR)]
    return df_out

num_cols = df_clean.drop("Outcome", axis=1).columns
df_clean = remove_outliers_iqr(df_clean, num_cols)

# =====================================================
#  FEATURE & NORMALIZATION
# =====================================================
X = df_clean.drop("Outcome", axis=1).values
y = df_clean["Outcome"].values

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

n_samples, n_features = X_norm.shape
N_CLUSTERS = 2

# =====================================================
#  BASELINE FCM
# =====================================================
cntr_fcm, u_fcm, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_norm.T, c=N_CLUSTERS, m=2.0, error=1e-5, maxiter=1000
)

labels_fcm = np.argmax(u_fcm.T, axis=1)
if accuracy_score(y, labels_fcm) < accuracy_score(y, 1 - labels_fcm):
    labels_fcm = 1 - labels_fcm

acc_fcm = accuracy_score(y, labels_fcm)
sens_fcm = sensitivity(y, labels_fcm)
spec_fcm = specificity(y, labels_fcm)
prec_fcm = precision_score(y, labels_fcm)

# =====================================================
#  PSO–FCM OPTIMIZATION
# =====================================================
DIM = N_CLUSTERS * n_features + 1

def decode_particle(p):
    centers = p[:-1].reshape(N_CLUSTERS, n_features)
    m = 1.5 + p[-1] * 2.0
    return centers, m

def pso_cost(swarm):
    costs = []
    for p in swarm:
        centers, m = decode_particle(p)
        costs.append(supervised_fitness(centers, X_norm, y, m))
    return np.array(costs)

optimizer = ps.single.GlobalBestPSO(
    n_particles=60,
    dimensions=DIM,
    options={"c1": 2.0, "c2": 2.0, "w": 0.7},
    bounds=(np.zeros(DIM), np.ones(DIM))
)

_, best_pos = optimizer.optimize(pso_cost, iters=200)
best_centers, best_m = decode_particle(best_pos)

# =====================================================
#  FINAL PSO-FCM LABELING
# =====================================================
diff = X_norm[:, None, :] - best_centers[None, :, :]
dist = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12

expo = 2.0 / (best_m - 1.0)
U = 1.0 / np.sum((dist[:, :, None] / dist[:, None, :]) ** expo, axis=2)

labels_pso = np.argmax(U, axis=1)
if accuracy_score(y, labels_pso) < accuracy_score(y, 1 - labels_pso):
    labels_pso = 1 - labels_pso

acc_pso = accuracy_score(y, labels_pso)
sens_pso = sensitivity(y, labels_pso)
spec_pso = specificity(y, labels_pso)
prec_pso = precision_score(y, labels_pso)

# =====================================================
#  HITUNG JUMLAH ANGGOTA CLUSTER
# =====================================================

# --- FCM ---
unique_fcm, count_fcm = np.unique(labels_fcm, return_counts=True)
cluster_count_fcm = dict(zip(unique_fcm, count_fcm))
cluster_count_fcm = {int(k): int(v) for k, v in cluster_count_fcm.items()}

# --- PSO-FCM ---
unique_pso, count_pso = np.unique(labels_pso, return_counts=True)
cluster_count_pso = dict(zip(unique_pso, count_pso))
cluster_count_pso = {int(k): int(v) for k, v in cluster_count_pso.items()}

# =====================================================
#  OUTPUT SUMMARY
# =====================================================
print("\n===================== EVALUATION SUMMARY =====================")
print(f"PSO-FCM Accuracy     : {acc_pso:.4f}")
print(f"PSO-FCM Sensitivity  : {sens_pso:.4f}")
print(f"PSO-FCM Specificity  : {spec_pso:.4f}")
print(f"PSO-FCM Precision    : {prec_pso:.4f}\n")

print(f"FCM Accuracy         : {acc_fcm:.4f}")
print(f"FCM Sensitivity      : {sens_fcm:.4f}")
print(f"FCM Specificity      : {spec_fcm:.4f}")
print(f"FCM Precision        : {prec_fcm:.4f}")

print("\n===================== CLUSTER SIZE =====================")
print("FCM Cluster Count    :", cluster_count_fcm)
print("PSO-FCM Cluster Count:", cluster_count_pso)

# =====================================================
# VISUALIZATION SECTION
# =====================================================

# PCA projection
pca = PCA(n_components=2)
X2 = pca.fit_transform(X_norm)
cent_pso_proj = pca.transform(best_centers)

plt.figure(figsize=(12,5))

# =====================================================
# 1) SCATTER PLOT: HASIL CLUSTER PSO-FCM
# =====================================================
plt.subplot(1,2,1)
plt.scatter(X2[:,0], X2[:,1],
            c=labels_pso,
            s=20,
            cmap='bwr',
            alpha=0.6)

plt.scatter(cent_pso_proj[:,0], cent_pso_proj[:,1],
            marker='x',
            s=200,
            linewidths=3,
            color='black')

plt.title("PSO-FCM Clustering (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")

# =====================================================
# 2) BAR CHART: SENSITIVITY & SPECIFICITY
# =====================================================
methods = ["FCM", "PSO-FCM"]
sens = [sens_fcm*100, sens_pso*100]
spec = [spec_fcm*100, spec_pso*100]

x = np.arange(len(methods))
width = 0.35

plt.subplot(1,2,2)
plt.bar(x - width/2, sens, width, label='Sensitivity')
plt.bar(x + width/2, spec, width, label='Specificity')

plt.xticks(x, methods)
plt.ylabel("Percentage (%)")
plt.ylim(0, 100)
plt.title("Sensitivity & Specificity Comparison")
plt.legend()

plt.tight_layout()
plt.show()
