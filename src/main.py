from data_preprocessing import TextPreprocessor, load_dataset
from markov_model import MarkovChain
from lstm_model import TextGeneratorLSTM
import numpy as np

def generer_texte_markov(texte: str, ordre: int = 2, nb_mots: int = 20):
    # Génère du texte avec Markov
    model = MarkovChain(order=ordre)
    model.train(texte)
    return model.generate(max_words=nb_mots)

def generer_texte_lstm(texte: str, taille_vocab: int = 100, longueur_seq: int = 10, nb_mots: int = 20):
    # Génère du texte avec LSTM
    preprocessor = TextPreprocessor(max_vocab_size=taille_vocab)
    cleaned_text = preprocessor.clean_text(texte)
    preprocessor.build_vocabulary(cleaned_text)
    sequences = preprocessor.text_to_sequences(cleaned_text)

    model = TextGeneratorLSTM(vocab_size=taille_vocab, sequence_length=longueur_seq)
    
    X = []
    y = np.zeros((len(sequences) - longueur_seq, taille_vocab))
    
    for i in range(len(sequences) - longueur_seq):
        seq = sequences[i:i + longueur_seq]
        target = sequences[i + longueur_seq]
        X.append(seq)
        y[i, target] = 1
    
    X = np.array(X)
    
    print("Entraînement du modèle LSTM...")
    model.train(X, y, epochs=10)
    
    seed_sequence = X[0]
    generated_indices = model.generate(seed_sequence, words_to_generate=nb_mots)
    
    return preprocessor.sequences_to_text(generated_indices)

def main():
    print("=== Générateur de Texte avec Markov et LSTM ===")
    
    texte = load_dataset('../data/dataset.txt')
    preprocessor = TextPreprocessor()
    texte_nettoye = preprocessor.clean_text(texte)
    
    print("\n1. Génération avec Markov")
    print("-" * 40)
    texte_markov = generer_texte_markov(texte_nettoye)
    print(texte_markov)
    
    print("\n2. Génération avec LSTM")
    print("-" * 40)
    texte_lstm = generer_texte_lstm(texte_nettoye)
    print(texte_lstm)

if __name__ == '__main__':
    main()
