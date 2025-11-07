# %%
# Import des bibliothèques nécessaires pour le projet
import zipfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
#%%
# Extraction de l'archive contenant le dataset
zip_path = "C:/ClassificationImage/archive.zip"
extract_path = "C:/ClassificationImage/rice_dataset/"

print(" Dézippage en cours")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print(" Dézippage terminé avec succès")

# Inspection du contenu dézippé
print("\n Contenu du dossier dézippé:")
for item in os.listdir(extract_path):
    item_path = os.path.join(extract_path, item)
    if os.path.isdir(item_path):
        print(f" Dossier '{item}' : {len(os.listdir(item_path))} éléments")
    else:
        print(f" Fichier: {item}")

# %% 
# Chargement et préparation des images du dataset
folder_path = os.path.join(extract_path, "rice_leaf_diseases")
classes = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]


def charger_dataset(chemin_dossier, classes_cibles):
    """
    Charge les images du dataset et les prépare pour l'entraînement.

    Arguments :
    - chemin_dossier : Chemin vers le dossier principal contenant les sous-dossiers de classes
    - classes_cibles : Liste des noms de classes à charger

    Retour :
    - Liste contenant les images et leurs labels correspondants
    """
    dataset = []
    
    for nom_classe in classes_cibles:
        chemin_classe = os.path.join(chemin_dossier, nom_classe)
        index_label = classes_cibles.index(nom_classe)
        
        print(f" Chargement de la classe '{nom_classe}': ", end="")
        
        # Filtrage des fichiers image valides
        fichiers_image = [f for f in os.listdir(chemin_classe) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        compteur = 0
        for fichier_image in tqdm(fichiers_image, desc=nom_classe):
            chemin_image = os.path.join(chemin_classe, fichier_image)
            image = cv2.imread(chemin_image)
            
            if image is not None:
                # Redimensionnement standard pour la cohérence
                image = cv2.resize(image, (224, 224))
                dataset.append([image, index_label])
                compteur += 1
        
        print(f" {compteur} images chargées")
    
    return dataset

# Chargement effectif du dataset
dataset = charger_dataset(folder_path, classes)
print(f"\n Dataset complet: {len(dataset)} images au total")

def preparer_donnees(dataset_brut):
    """
    Prépare les données pour l'entraînement du modèle.

    Arguments :
    - dataset_brut : Liste brute des images et labels

    Retour :
    - X : Tableau numpy des images normalisées
    - y : Tableau numpy des labels
    """
    images_X = []
    labels_y = []
    
    for image, label in dataset_brut:
        images_X.append(image)
        labels_y.append(label)
    
    # Normalisation des valeurs de pixels entre 0 et 1
    X_normalise = np.array(images_X) / 255.0
    y_normalise = np.array(labels_y)
    
    return X_normalise, y_normalise

X, y = preparer_donnees(dataset)

print(f" Données préparées avec succès")
print(f"   Forme de X (images): {X.shape}")
print(f"   Forme de y (labels): {y.shape}")

def visualiser_echantillons(images, labels, noms_classes, nombre_echantillons=6):
    """
    Affiche un échantillon d'images du dataset avec leurs labels.

    Arguments :
    - images : Tableau des images à afficher
    - labels : Tableau des labels correspondants
    - noms_classes : Liste des noms des classes
    - nombre_echantillons : Nombre d'images à afficher

    Retour :
    - Aucun
    """
    print(f"\n Visualisation de {nombre_echantillons} échantillons du dataset:")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(nombre_echantillons):
        index_aleatoire = np.random.randint(len(images))
        ax = axes[i//3, i%3]
        
        # Conversion BGR vers RGB pour un affichage correct
        image_rgb = cv2.cvtColor((images[index_aleatoire] * 255).astype(np.uint8), 
                                cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.set_title(f"Classe: {noms_classes[labels[index_aleatoire]]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

visualiser_echantillons(X, y, classes)

# %% 
# Division des données en ensembles d'entraînement et de test

def diviser_donnees(images, labels, taille_test=0.2, graine_aleatoire=42):
    """
    Divise le dataset en ensembles d'entraînement et de test.

    Arguments :
    - images : Tableau des images
    - labels : Tableau des labels
    - taille_test : Proportion des données à utiliser pour le test
    - graine_aleatoire : Seed pour la reproductibilité

    Retour :
    - X_train, X_test, y_train, y_test : Les ensembles divisés
    """
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, 
        test_size=taille_test, 
        random_state=graine_aleatoire, 
        stratify=labels
    )
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = diviser_donnees(X, y)

print(f" Division train/test effectuée:")
print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")

def creer_modele_cnn(taille_entree=(224, 224, 3), nombre_classes=3):
    """
    Crée un modèle CNN simple pour la classification d'images.

    Arguments :
    - taille_entree : Dimensions des images d'entrée
    - nombre_classes : Nombre de classes de sortie

    Retour :
    - model : Modèle Keras compilé
    """
    modele = Sequential([
        # Première couche de convolution avec pooling
        Conv2D(32, (3, 3), activation='relu', input_shape=taille_entree),
        MaxPooling2D((2, 2)),
        
        # Deuxième couche de convolution avec plus de filtres
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Troisième couche de convolution pour capturer des features complexes
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Couches fully connected pour la classification
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Réduction du surapprentissage
        Dense(nombre_classes, activation='softmax')  # Sortie pour 3 classes
    ])
    
    # Compilation du modèle
    modele.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return modele

# Création du modèle de base
model = creer_modele_cnn()

print(" Architecture du modèle CNN de base:")
model.summary()

# %%
# Entraînement du modèle de base

def entrainer_modele(modele, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    """
    Entraîne le modèle et retourne l'historique d'entraînement.

    Arguments :
    - modele : Modèle Keras à entraîner
    - X_train, y_train : Données d'entraînement
    - X_test, y_test : Données de validation
    - epochs : Nombre d'epochs d'entraînement
    - batch_size : Taille des lots

    Retour :
    - history : Historique contenant les métriques d'entraînement
    """
    history = modele.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return history

print(" Début de l'entraînement du modèle de base")
history = entrainer_modele(model, X_train, y_train, X_test, y_test)
print(" Entraînement terminé")


# Sauvegarde du modèle entraîné
model.save('modele_riz_conv2d.h5')
print(" Modèle sauvegardé sous: 'modele_riz_conv2d.h5'")

def visualiser_performances(historique):
    """
    Visualise les courbes d'apprentissage du modèle.

    Arguments :
    - historique : Historique retourné par model.fit()

    Retour :
    - Aucun
    """
    plt.figure(figsize=(12, 4))
    
    # Courbe d'accuracy
    plt.subplot(1, 2, 1)
    plt.plot(historique.history['accuracy'], label='Accuracy Entraînement', linewidth=2)
    plt.plot(historique.history['val_accuracy'], label='Accuracy Validation', linewidth=2)
    plt.title('Évolution de l\'Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Courbe de loss
    plt.subplot(1, 2, 2)
    plt.plot(historique.history['loss'], label='Loss Entraînement', linewidth=2)
    plt.plot(historique.history['val_loss'], label='Loss Validation', linewidth=2)
    plt.title('Évolution de la Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualiser_performances(history)

print(f"\n Performances du modèle de base:")
print(f" Meilleure accuracy entraînement: {max(history.history['accuracy']):.2%}")
print(f" Meilleure accuracy validation: {max(history.history['val_accuracy']):.2%}")

# Évaluation du modèle
perte_test, accuracy_test = model.evaluate(X_test, y_test, verbose=0)
print(f" Accuracy sur l'ensemble de test: {accuracy_test:.2%}")

# %% 
# Création d'un modèle amélioré avec régularisation

def creer_modele_ameliore(taille_entree=(224, 224, 3), nombre_classes=3):
    """
    Crée un modèle CNN amélioré avec régularisation pour réduire le surapprentissage.

    Arguments :
    - taille_entree : Dimensions des images d'entrée
    - nombre_classes : Nombre de classes de sortie

    Retour :
    - model : Modèle Keras amélioré et compilé
    """
    modele = Sequential([
        # Première couche avec régularisation L2 et dropout
        Conv2D(32, (3, 3), activation='relu', 
               input_shape=taille_entree, 
               kernel_regularizer=l2(0.01)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),  # Dropout précoce pour la régularisation
        
        # Deuxième couche
        Conv2D(64, (3, 3), activation='relu', 
               kernel_regularizer=l2(0.01)),
        MaxPooling2D((2, 2)),
        Dropout(0.3),  # Dropout augmenté progressivement
        
        # Troisième couche
        Conv2D(128, (3, 3), activation='relu', 
               kernel_regularizer=l2(0.01)),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        # Couches fully connected avec régularisation
        Flatten(),
        Dense(128, activation='relu', 
              kernel_regularizer=l2(0.01)),
        Dropout(0.5),  # Fort dropout sur les couches denses
        Dense(nombre_classes, activation='softmax')
    ])
    
    return modele

modele_amel = creer_modele_ameliore()

# %% 
# Data Augmentation pour améliorer la généralisation

def configurer_augmentation_donnees():
    """
    Configure le générateur de data augmentation pour enrichir le dataset.

    Retour :
    - datagen : Générateur ImageDataGenerator configuré
    """
    datagen = ImageDataGenerator(
        rotation_range=20,      # Rotation aléatoire jusqu'à 20 degrés
        width_shift_range=0.2,  # Déplacement horizontal aléatoire
        height_shift_range=0.2, # Déplacement vertical aléatoire
        horizontal_flip=True,   # Retournement horizontal aléatoire
        zoom_range=0.2,         # Zoom aléatoire
        shear_range=0.2,        # Cisaillement aléatoire
        fill_mode='nearest'     # Méthode de remplissage pour les transformations
    )
    
    return datagen

datagen = configurer_augmentation_donnees()

# Configuration de l'optimiseur avec learning rate personnalisé
optimizer = Adam(learning_rate=0.001)
modele_amel.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(" Début de l'entraînement avec data augmentation")
history_amel = modele_amel.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    steps_per_epoch=len(X_train) // 32,
    validation_data=(X_test, y_test)
)
print(" Entraînement avec augmentation terminé")

# %% 
# Évaluation détaillée du modèle amélioré

def evaluer_modele_complet(modele, X_test, y_test, noms_classes):
    """
    Évalue le modèle de manière complète avec diverses métriques.

    Arguments :
    - modele : Modèle Keras à évaluer
    - X_test, y_test : Données de test
    - noms_classes : Noms des classes pour l'affichage

    Retour :
    - Aucun
    """
    # Prédictions sur l'ensemble de test
    predictions = modele.predict(X_test)
    classes_predites = np.argmax(predictions, axis=1)
    
    # Rapport de classification détaillé
    print("\n" + "="*60)
    print(" RAPPORT DE CLASSIFICATION")
    print("="*60)
    print(classification_report(y_test, classes_predites, 
                              target_names=noms_classes))
    
    # Matrice de confusion
    plt.figure(figsize=(8, 6))
    matrice_confusion = confusion_matrix(y_test, classes_predites)
    sns.heatmap(matrice_confusion, annot=True, fmt='d', cmap='Blues', 
                xticklabels=noms_classes, yticklabels=noms_classes)
    plt.title('Matrice de Confusion')
    plt.ylabel('Labels Réels')
    plt.xlabel('Labels Prédits')
    plt.show()
    
    # Accuracy par classe pour une analyse détaillée
    print("\n ACCURACY PAR CLASSE:")
    print("=" * 30)
    for i, classe in enumerate(noms_classes):
        accuracy_classe = np.mean(classes_predites[y_test == i] == i)
        print(f" {classe:25}: {accuracy_classe:.2%}")

evaluer_modele_complet(modele_amel, X_test, y_test, classes)

# %% 
# Visualisation comparative des performances

def comparer_modeles(historique_base, historique_ameliore):
    """
    Compare visuellement les performances des deux modèles.

    Arguments :
    - historique_base : Historique du modèle de base
    - historique_ameliore : Historique du modèle amélioré

    Retour :
    - Aucun
    """
    plt.figure(figsize=(15, 5))
    
    # Comparaison Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(historique_ameliore.history['accuracy'], 
             label='Train (Avec Augmentation)', linewidth=2, color='blue')
    plt.plot(historique_ameliore.history['val_accuracy'], 
             label='Validation (Avec Augmentation)', linewidth=2, color='red')
    plt.plot(historique_base.history['accuracy'], 
             label='Train (Base)', linewidth=2, color='blue', alpha=0.3, linestyle='--')
    plt.plot(historique_base.history['val_accuracy'], 
             label='Validation (Base)', linewidth=2, color='red', alpha=0.3, linestyle='--')
    plt.title('Comparaison Accuracy: Base vs Amélioré\n(Réduction du surapprentissage visé)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparaison Loss
    plt.subplot(1, 2, 2)
    plt.plot(historique_ameliore.history['loss'], 
             label='Train (Avec Augmentation)', linewidth=2, color='blue')
    plt.plot(historique_ameliore.history['val_loss'], 
             label='Validation (Avec Augmentation)', linewidth=2, color='red')
    plt.plot(historique_base.history['loss'], 
             label='Train (Base)', linewidth=2, color='blue', alpha=0.3, linestyle='--')
    plt.plot(historique_base.history['val_loss'], 
             label='Validation (Base)', linewidth=2, color='red', alpha=0.3, linestyle='--')
    plt.title('Comparaison Loss: Base vs Amélioré')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

comparer_modeles(history, history_amel)

# %%
# Synthèse finale des performances

print(f"\n" + "="*50)
print(" SYNTHÈSE DES PERFORMANCES FINALES")
print("="*50)
print(f" Modèle de base - Meilleure accuracy validation: {max(history.history['val_accuracy']):.2%}")
print(f" Modèle amélioré - Meilleure accuracy validation: {max(history_amel.history['val_accuracy']):.2%}")

amelioration = max(history_amel.history['val_accuracy']) - max(history.history['val_accuracy'])
print(f" Amélioration: {amelioration:+.2%}")

if amelioration > 0:
    print(" Le modèle amélioré performe mieux!")
else:
    print(" Le modèle de base reste meilleur")
# %%
