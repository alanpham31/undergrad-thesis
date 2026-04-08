import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from pathlib import Path

# scikit-learn tools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, log_loss, roc_curve, auc
from sklearn.feature_selection import VarianceThreshold

# RDKit & Mordred (for Pyrfume features)
from rdkit import Chem
from mordred import Calculator, descriptors

DATA_DIR = Path("data")

# load data and prep
print("loading data")
qualities  = pd.read_csv(DATA_DIR / "keller_qualities.csv")
ir_spectra = pd.read_csv(DATA_DIR / "ir_spectra_matrix.csv", index_col=0)
molecules  = pd.read_csv(DATA_DIR / "molecules.csv")

# filter for high concentration only
qualities = qualities[qualities['Concentration'] == 0.001].copy()

# find dominant descriptor (target y for classification)
# drop metadata columns to isolate 20 descriptor ratings
desc_cols = [c for c in qualities.columns if c not in ['Stimulus', 'CID', 'Concentration']]
# find column name with highest value for each row
qualities['dominant_percept'] = qualities[desc_cols].idxmax(axis=1)

# merge percepts with IR spectra
df_merged = pd.merge(qualities[['CID', 'dominant_percept']], ir_spectra, on="CID", how='inner')

# find intersection of CIDs
valid_cids = df_merged['CID'].unique()
print(f"Total overlapping high-concentration molecules: {len(valid_cids)}")


# exploratory analysis (PCA and clustering)
print("---exploratory analysis---")

# isolate IR features 
X_ir_raw = df_merged.drop(columns=['CID', 'dominant_percept'])

# standardize before PCA
scaler_ir = StandardScaler()
X_ir_scaled = scaler_ir.fit_transform(X_ir_raw)

# PCA on IR spectra
pca = PCA(n_components = 2)
ir_pca = pca.fit_transform(X_ir_scaled)

# cluster IR spectra
kmeans = KMeans(n_clusters=5, random_state=42)
ir_clusters = kmeans.fit_predict(X_ir_scaled)

# plotting - paint perceptual labels onto IR PCA
plt.figure(figsize=(12, 8))
sns.scatterplot(x=ir_pca[:, 0], y=ir_pca[:, 1], hue=df_merged['dominant_percept'], style=ir_clusters, palette='tab20', s=100)
plt.title("PCA of IR spectra colored by dominant perceptual descriptor")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100}%)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# feature engineering (mordred / pyrfume)
print("---calculating mordred features---")
# filter molecules
mols_subset = molecules[molecules['CID'].isin(valid_cids)].copy()

# initialize mordred calc
calc = Calculator(descriptors, ignore_3D=True) 
# 2D features are faster and more robust


def get_mordred(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            return list(calc(mol))
        except:
            pass
    return [0] * len(calc.descriptors)


# calc features
features_list = mols_subset['CanonicalSMILES'].apply(get_mordred)
mordred_df = pd.DataFrame(features_list.tolist(), index=mols_subset['CID'])
mordred_df = mordred_df.apply(pd.to_numeric, errors='coerce').fillna(0) # to clean errors
mordred_df = mordred_df.reset_index()

# merge into master dataframe
master_df = pd.merge(df_merged, mordred_df, on='CID', how='inner')


# prediction task (classification showdown)
print("---classification task---")

# check distribution of classes
class_counts = master_df['dominant_percept'].value_counts()
print("original class distribution: ", class_counts)

# filter rare classes (classes must have at least 5 examples)
min_samples = 5
valid_classes = class_counts[class_counts >= min_samples].index

# apply filter
master_df = master_df[master_df['dominant_percept'].isin(valid_classes)].copy()
print(f"dropped {len(class_counts) - len(valid_classes)} rare classes")

# encode target labels to ints
le = LabelEncoder()
y = le.fit_transform(master_df['dominant_percept'])
print(y.shape)

# define feature spaces
X_ir       = master_df[X_ir_raw.columns]
X_shape    = master_df[mordred_df.columns.drop('CID')]
X_combined = pd.concat([X_shape, X_ir], axis=1)


def evaluate_scenario(X, y, scenario_name, k_features=50):
    print(f"evaluating scenario: {scenario_name}")

    X.columns = X.columns.astype(str)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # remove constant features (errors with UserWarning and RuntimeWarnings)
    selector_var = VarianceThreshold(threshold=0.0)
    X_train_v = selector_var.fit_transform(X_train)
    X_test_v  = selector_var.transform(X_test)

    # standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_v)
    X_test_s  = scaler.transform(X_test_v)
    
    # feature selection
    # make sure k_features aren't bigger than remaining valid features
    k_actual = min(k_features, X_train_s.shape[1])
    selector  = SelectKBest(score_func=f_classif, k=k_actual)
    X_train_k = selector.fit_transform(X_train_s, y_train)
    X_test_k  = selector.transform(X_test_s)

    # model: random forest (calc cross-entropy with predict_proba)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_k, y_train)

    # predictions
    y_pred  = clf.predict(X_test_k)
    y_proba = clf.predict_proba(X_test_k)

    # metrics
    loss = log_loss(y_test, y_proba)
    # AUROC needs one v rest for multi class
    try:
        auroc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except ValueError:
        auroc = np.nan # handles edge cases where class is missing in test set


    print(f"cross-entropy loss: {loss:.4f}")
    print(f"AUROC (one v rest): {auroc:.4f}")

    # classification results (precision, recall, f1)
    print("---classification results---")
    target_names = le.inverse_transform(np.unique(y))
    print(classification_report(y_test, y_pred, target_names=le.inverse_transform(np.unique(y)), zero_division=0))

    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"confusion matrix: {scenario_name}")
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.tight_layout()
    plt.show()


# run 3 scenarios
evaluate_scenario(X_shape, y, "1. Physicochemical Only (Mordred)", k_features=50)
evaluate_scenario(X_ir, y, "2. Spectral Data Only (IR)", k_features=50)
evaluate_scenario(X_combined, y, "3. Combined (Mordred + IR)", k_features=100)

'''
# 2. View the first 5 SMILES strings and their corresponding odor labels
print("--- Raw Data ---")
print(df[['SMILES', 'dominant_percept']].head())

# 3. Convert a SMILES string into an RDKit Molecule object
# Let's take the first molecule in your dataset
sample_smiles = df['SMILES'].iloc[0]
mol = Chem.MolFromSmiles(sample_smiles)

# 4. Verify the molecule was created successfully
if mol is not None:
    print(f"\nSuccess! Converted SMILES '{sample_smiles}' into a 2D molecule.")
    print(f"This molecule has {mol.GetNumAtoms()} heavy atoms.")
else:
    print("\nError: Invalid SMILES string.")
'''

# Poster styling setup
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("poster", font_scale=0.8)
colors = ['#4A90E2', '#E94B3C', '#50C878'] # Blue, Red, Green

def extract_model_metrics(X, y, k_features=50):
    """Helper function to run the pipeline and return exact metrics for plotting."""
    X.columns = X.columns.astype(str)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Variance threshold
    selector_var = VarianceThreshold(threshold=0.0)
    X_train_v = selector_var.fit_transform(X_train)
    X_test_v  = selector_var.transform(X_test)
    
    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_v)
    X_test_s  = scaler.transform(X_test_v)
    
    # Feature selection
    k_actual = min(k_features, X_train_s.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k_actual)
    X_train_k = selector.fit_transform(X_train_s, y_train)
    X_test_k  = selector.transform(X_test_s)
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_k, y_train)
    
    # Probabilities & AUROC
    y_proba = clf.predict_proba(X_test_k)
    try:
        auroc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except ValueError:
        auroc = np.nan
        
    # Extract the names of the features that survived the SelectKBest
    surviving_var_features = X.columns[selector_var.get_support()]
    final_feature_names = surviving_var_features[selector.get_support()]
        
    return auroc, y_test, y_proba, clf, final_feature_names

print("\n--- Extracting Real Data for Poster Graphs ---")
# 1. Extract metrics for all three scenarios
auroc_shape, y_test_shape, proba_shape, _, _ = extract_model_metrics(X_shape, y, k_features=50)
auroc_ir, y_test_ir, proba_ir, _, _ = extract_model_metrics(X_ir, y, k_features=50)
auroc_comb, y_test_comb, proba_comb, clf_comb, feat_names_comb = extract_model_metrics(X_combined, y, k_features=100)

# ==========================================
# GRAPH 1: AUROC MODEL COMPARISON (BAR CHART)
# ==========================================
print("Generating AUROC Comparison Chart...")
models = ['Shape-Only\n(Mordred)', 'Shake-Only\n(IR Spectra)', 'Combined\n(Shape + Shake)']
auroc_scores = [auroc_shape, auroc_ir, auroc_comb] 

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(models, auroc_scores, color=colors, edgecolor='black', linewidth=1.5)

for bar in bars:
    yval = bar.get_height()
    if not np.isnan(yval):
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', 
                ha='center', va='bottom', fontweight='bold')

ax.set_ylim(0, 1.0)
ax.set_ylabel('AUROC Score (OVR)', fontweight='bold')
ax.set_title('Overall Predictive Performance Across Models', fontweight='bold', pad=20)
plt.savefig('auroc_comparison_poster.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# GRAPH 2: ROC CURVES FOR A SPECIFIC DESCRIPTOR
# ==========================================
print("Generating ROC Curves...")
# Find the index of the most common valid class to plot
target_class_idx = pd.Series(y_test_comb).value_counts().idxmax()
target_class_name = le.inverse_transform([target_class_idx])[0]

# Binarize the true labels for the specific class (1 if target class, 0 otherwise)
y_true_binary = (y_test_comb == target_class_idx).astype(int)

# Get the probabilities for just that class
prob_shape_class = proba_shape[:, target_class_idx]
prob_ir_class = proba_ir[:, target_class_idx]
prob_comb_class = proba_comb[:, target_class_idx]

fig, ax = plt.subplots(figsize=(8, 8))

# Shape
fpr_shape, tpr_shape, _ = roc_curve(y_true_binary, prob_shape_class)
ax.plot(fpr_shape, tpr_shape, color=colors[0], lw=3, label=f'Shape (AUC = {auc(fpr_shape, tpr_shape):.2f})')

# Shake (IR)
fpr_ir, tpr_ir, _ = roc_curve(y_true_binary, prob_ir_class)
ax.plot(fpr_ir, tpr_ir, color=colors[1], lw=3, label=f'Shake (AUC = {auc(fpr_ir, tpr_ir):.2f})')

# Combined
fpr_comb, tpr_comb, _ = roc_curve(y_true_binary, prob_comb_class)
ax.plot(fpr_comb, tpr_comb, color=colors[2], lw=3, label=f'Combined (AUC = {auc(fpr_comb, tpr_comb):.2f})')

ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title(f'ROC Curves: Predicting "{target_class_name}" Odor', fontweight='bold', pad=20)
ax.legend(loc="lower right", frameon=True, shadow=True)
plt.savefig('roc_curve_specific_poster.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# GRAPH 3: TOP 10 FEATURE IMPORTANCE 
# ==========================================
print("Generating Feature Importance Chart...")
# Get importances and tie them to feature names
importances = clf_comb.feature_importances_
feature_importance_dict = dict(zip(feat_names_comb, importances))

# Sort to get top 10
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
top_feat_names = [x[0] for x in sorted_features]
top_feat_scores = [x[1] for x in sorted_features]

fig, ax = plt.subplots(figsize=(10, 6))

# Color code based on origin (Shape vs Shake)
bar_colors = []
for feat in top_feat_names:
    if feat in X_shape.columns:
        bar_colors.append(colors[0]) # Blue for Shape
    else:
        bar_colors.append(colors[1]) # Red for Shake

sns.barplot(x=top_feat_scores, y=top_feat_names, palette=bar_colors, edgecolor='black', ax=ax)

ax.set_xlabel('Random Forest Feature Importance Score', fontweight='bold')
ax.set_title('Top 10 Predictive Features (Combined Model)', fontweight='bold', pad=20)

legend_elements = [Patch(facecolor=colors[0], edgecolor='black', label='Shape Feature (Mordred)'),
                   Patch(facecolor=colors[1], edgecolor='black', label='Shake Feature (IR)')]
ax.legend(handles=legend_elements, loc='lower right')
plt.savefig('feature_importance_poster.png', dpi=300, bbox_inches='tight')
plt.close()

print("All poster graphs successfully generated and saved as 300 DPI PNG files based on your processed data!")