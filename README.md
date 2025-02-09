# Générateur de Texte - Projet IA

Ce projet implémente deux approches différentes pour la génération de texte : un modèle basé sur les chaînes de Markov et un modèle utilisant des réseaux de neurones LSTM (Long Short-Term Memory).

## Table des Matières

1. [Prérequis](#prérequis)
2. [Installation](#installation)
3. [Structure du Projet](#structure-du-projet)
4. [Utilisation](#utilisation)
5. [Description des Modèles](#description-des-modèles)
6. [Résultats](#résultats)

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

## Structure du Projet

```
GENIA2/
│
├── data/
│   └── dataset.txt          # Texte d'entraînement (Contes de Grimm)
│
├── src/
│   ├── data_preprocessing.py # Prétraitement des données
│   ├── markov_model.py      # Implémentation du modèle de Markov
│   ├── lstm_model.py        # Implémentation du modèle LSTM
│   └── main.py             # Script principal
│
├── README.md
└── requirements.txt
```

## Utilisation

1. Pour lancer le programme principal qui compare les deux modèles :
```bash
cd src
python main.py
```

2. Pour utiliser uniquement le modèle de Markov :
```python
from markov_model import MarkovChain
from data_preprocessing import load_dataset, TextPreprocessor

# Charger et prétraiter les données
texte = load_dataset('../data/dataset.txt')
preprocessor = TextPreprocessor()
texte_nettoye = preprocessor.clean_text(texte)

# Créer et entraîner le modèle
model = MarkovChain(order=2)
model.train(texte_nettoye)

# Générer du texte
texte_genere = model.generate(max_words=20)
print(texte_genere)
```

3. Pour utiliser uniquement le modèle LSTM :
```python
from lstm_model import TextGeneratorLSTM
from data_preprocessing import load_dataset, TextPreprocessor

# Charger et prétraiter les données
texte = load_dataset('../data/dataset.txt')
preprocessor = TextPreprocessor(max_vocab_size=100)
texte_nettoye = preprocessor.clean_text(texte)

# Créer et entraîner le modèle
model = TextGeneratorLSTM(vocab_size=100, sequence_length=10)
# Voir main.py pour l'implémentation complète de l'entraînement
```

## Description des Modèles

### Modèle de Markov
- Utilise les probabilités de transition entre les mots
- Implémente une chaîne de Markov d'ordre 2 (considère les 2 mots précédents)
- Avantages : rapide, simple, bon pour les motifs locaux
- Limitations : pas de compréhension du contexte global

### Modèle LSTM
- Utilise des réseaux de neurones récurrents
- Architecture : Embedding + LSTM + Dense
- Avantages : meilleure compréhension du contexte, génération plus cohérente
- Limitations : nécessite plus de données et de temps d'entraînement

## Résultats

Les deux modèles produisent des résultats différents :

1. Modèle de Markov :
   - Génération rapide
   - Bonne cohérence locale
   - Peut reproduire des phrases du texte d'origine

2. Modèle LSTM :
   - Génération plus créative
   - Meilleure structure grammaticale
   - Nécessite plus d'entraînement pour de bons résultats

## Améliorations Possibles

1. Augmenter la taille du jeu de données
2. Ajuster les hyperparamètres des modèles
3. Implémenter une interface utilisateur graphique
4. Ajouter plus de métriques d'évaluation
5. Expérimenter avec différentes architectures LSTM
#   G E N I A E P S I  
 