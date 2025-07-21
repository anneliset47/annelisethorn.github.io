import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned Ensembl dataset
csv_path = '/Users/annelisethorn/Documents/School/Summer 2025/Machine Learning/Datasets/cleaned_ensembl.csv'
ensembl_df = pd.read_csv(csv_path)

# Select features and sample down for memory efficiency
features = ['Length', 'strand']
X = ensembl_df[features].dropna().sample(n=1000, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try multiple values of k
k_values = [2, 3, 4, 5]
silhouette_scores = {}
kmeans_results = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores[k] = score
    kmeans_results.append((k, labels))

# Identify best k
best_k = max(silhouette_scores, key=silhouette_scores.get)
best_labels = [labels for k, labels in kmeans_results if k == best_k][0]
X['Cluster'] = best_labels

# Save clustered data
X.to_csv("kmeans_clustered_data.csv", index=False)

# Plot silhouette scores
silhouette_df = pd.DataFrame(list(silhouette_scores.items()), columns=["k", "SilhouetteScore"])
plt.figure(figsize=(6, 4))
sns.lineplot(data=silhouette_df, x='k', y='SilhouetteScore', marker='o')
plt.title("Silhouette Scores for Different k")
plt.tight_layout()
plt.savefig("silhouette_scores_sampled.png")
