import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

# Charger les jeux de données
x_train = pd.read_csv(r"C:\Users\victo\Desktop\IMTatlantique\data_challenge\x_train_final.csv")
y_train = pd.read_csv(r"C:\Users\victo\Desktop\IMTatlantique\data_challenge\y_train_final_j5KGWWK.csv")
x_test = pd.read_csv(r"C:\Users\victo\Desktop\IMTatlantique\data_challenge\x_test_final.csv")  # Fichier test

# Ajouter y_train['p0q0'] à x_train
x_train['p0q0'] = y_train['p0q0']

# Sélection des variables
features = ['train', 'gare', 'arret', 'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4','p0q0']
features2 = ['p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4','p0q0']


# Initialiser l'encoder avec gestion des valeurs inconnues
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

# Appliquer l'encodage sur train et test
x_train[['train', 'gare']] = encoder.fit_transform(x_train[['train', 'gare']])
x_train['date'] = pd.to_datetime(x_train['date']).astype(int) / 10**9  # Convertir en timestamp (secondes)

#==============================================================================

# On utilise "spearman" ou "pearson" pour la corrélation comme argument de la méthode corr()
# Calcul de la matrice de corrélation
corr = x_train[features].corr('spearman') # Corrélation de Spearman 
corr2 = x_train[features2].corr("spearman") # Corrélation de Spearman

#==============================================================================

# Affichage de la matrice de corrélation totale ( avec train, date, et arret) sous forme de heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de corrélation')
plt.show()

# Affichage de la matrice de corrélation réduite ( sans train, date, et arret) sous forme de heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr2, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de corrélation')
plt.show()


