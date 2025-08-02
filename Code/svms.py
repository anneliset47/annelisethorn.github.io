import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load datasets
clinvar_df = pd.read_csv('/Users/annelisethorn/Documents/School/Summer 2025/Machine Learning/Datasets/clinvar_repeat_pathogenic_variants.csv')
ensembl_df = pd.read_csv('/Users/annelisethorn/Documents/School/Summer 2025/Machine Learning/Datasets/ensembl_tandem_repeats.csv')
populations_df = pd.read_csv('/Users/annelisethorn/Documents/School/Summer 2025/Machine Learning/Datasets/igsr-1kg_ont_vienna-populations.tsv', sep='\t')
samples_df = pd.read_csv('/Users/annelisethorn/Documents/School/Summer 2025/Machine Learning/Datasets/igsr-1kg_ont_vienna-samples.tsv', sep='\t')

# Labeling
clinvar_df['label'] = clinvar_df['ClinicalSignificance'].apply(lambda x: 1 if 'Pathogenic' in x else 0)

# Feature engineering
clinvar_df['gene_length'] = clinvar_df['Gene'].apply(lambda x: len(str(x)))
clinvar_df['title_length'] = clinvar_df['Title'].apply(lambda x: len(str(x)))
gene_encoder = LabelEncoder()
clinvar_df['gene_encoded'] = gene_encoder.fit_transform(clinvar_df['Gene'].astype(str))

# Define features and labels
features = clinvar_df[['gene_length', 'title_length', 'gene_encoded']]
labels = clinvar_df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Train SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluate
y_pred = svm_model.predict(X_test)
report = classification_report(y_test, y_pred)
print("SVM Classification Report:")
print(report)
 