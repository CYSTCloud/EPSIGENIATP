## Questions et Réponses
  ### Qu'est-ce qu'un modèle génératif de texte et comment fonctionne-t-il ?

Un modèle génératif de texte est un système qui peut créer du nouveau texte en se basant sur ce qu'il a appris à partir d'exemples. C'est un peu comme quand on apprend à écrire en lisant beaucoup de livres.

Le fonctionnement se fait en deux étapes principales :
1. **Phase d'apprentissage** : Le modèle analyse un grand nombre de textes pour comprendre :
   - Comment les mots se suivent
   - Les structures des phrases
   - Les patterns du langage

2. **Phase de génération** : Le modèle utilise ce qu'il a appris pour créer du nouveau texte :
   - Il commence avec un mot ou une phrase
   - Il prédit le mot suivant le plus probable
   - Il continue mot par mot jusqu'à avoir un texte complet

#### 2. Quelle est la différence entre un modèle probabiliste et un modèle neuronal ?

**Modèle probabiliste (Markov) :**
- Plus simple à comprendre et à implémenter
- Fonctionne comme un dé : il choisit le prochain mot selon des probabilités fixes
- Exemple : Si j'ai écrit "il fait", le modèle regarde juste quels mots suivent souvent "fait"

**Modèle neuronal (LSTM) :**
- Fonctionne comme un cerveau avec une mémoire
- Se souvient du contexte sur une longue période
- Comprend mieux les relations entre les mots
- Exemple : Si j'écris "Pierre a pris son parapluie car...", le modèle comprend qu'il va probablement pleuvoir

#### 3. Dans quels domaines les modèles génératifs sont-ils utilisés ?

Les modèles génératifs ont plein d'applications cool :
- **Création de contenu** : 
  - Écriture d'articles
  - Génération de poésie
  - Création de dialogues pour les jeux vidéo
  
- **Aide à l'écriture** :
  - Complétion automatique
  - Correction grammaticale
  - Suggestions de reformulation

- **Applications pratiques** :
  - Chatbots
  - Traduction automatique
  - Résumé automatique de textes

#### 4. Quels défis peuvent apparaître dans la génération de texte automatique ?

Il y a plusieurs défis importants à surmonter :

1. **Cohérence** :
   - Garder un sens logique sur plusieurs phrases
   - Ne pas se contredire
   - Maintenir le même sujet

2. **Qualité linguistique** :
   - Respecter la grammaire
   - Utiliser le bon vocabulaire
   - Garder un style naturel

3. **Limitations techniques** :
   - Besoin de beaucoup de données d'entraînement
   - Temps de calcul important
   - Mémoire limitée pour le contexte

4. **Problèmes éthiques** :
   - Risque de biais dans les données
   - Génération de fausses informations
   - Questions de propriété intellectuelle

### Comment améliorer la qualité de la génération avec Markov ?

Plusieurs approches peuvent être utilisées pour améliorer la génération :

1. Filtrage intelligent : Implémenter des règles pour éviter les transitions incohérentes.
2. Prétraitement des données : Améliorer la qualité du corpus d'entraînement.

## Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Un environnement virtuel (recommandé)

## Installation

1. Clonez le dépôt :
```bash
git clone <url-du-depot>
cd GENIA2
```

2. Créez et activez un environnement virtuel (recommandé) :
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```
