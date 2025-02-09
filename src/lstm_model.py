import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextGeneratorLSTM:
    def __init__(self, vocab_size, sequence_length=10, embedding_dim=50):
        # Initialisation du modèle LSTM
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.model = self._build_model()

    def _build_model(self):
        # Construction du modèle LSTM
        model = Sequential([
            # Couche d'embedding pour convertir les indices de mots en vecteurs denses
            Embedding(self.vocab_size, self.embedding_dim, 
                     input_length=self.sequence_length),
            
            # Première couche LSTM avec retour de séquences pour empilage
            LSTM(128, return_sequences=True),
            
            # Deuxième couche LSTM
            LSTM(64),
            
            # Couche de sortie avec activation softmax
            Dense(self.vocab_size, activation='softmax')
        ])
        
        # Compiler le modèle avec la fonction de perte d'entropie croisée catégorielle
        model.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
        
        return model

    def prepare_sequences(self, text_sequences):
        # Préparation des séquences pour l'entraînement
        X = []
        y = []
        
        for sequence in text_sequences:
            for i in range(len(sequence) - self.sequence_length):
                X.append(sequence[i:i + self.sequence_length])
                y.append(sequence[i + self.sequence_length])
        
        X = np.array(X)
        y = tf.keras.utils.to_categorical(y, num_classes=self.vocab_size)
        
        return X, y

    def train(self, X, y, epochs=50, batch_size=32):
        # Entraînement du modèle
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def generate(self, seed_sequence, words_to_generate=20, temperature=1.0):
        # Génération de texte
        generated = list(seed_sequence)
        
        # Générer un mot à la fois
        for _ in range(words_to_generate):
            # Préparer la séquence d'entrée
            sequence = np.array([generated[-self.sequence_length:]])
            
            # Obtenir les prédictions du modèle
            predictions = self.model.predict(sequence, verbose=0)[0]
            
            # Appliquer l'échelle de température
            predictions = np.log(predictions) / temperature
            exp_predictions = np.exp(predictions)
            predictions = exp_predictions / np.sum(exp_predictions)
            
            # Échantillonner l'indice du mot suivant
            next_index = np.random.choice(len(predictions), p=predictions)
            
            # Ajouter à la séquence générée
            generated.append(next_index)
        
        return generated[self.sequence_length:]  # Retourner uniquement les mots nouvellement générés

def main():
    # Exemple d'utilisation
    from data_preprocessing import TextPreprocessor, load_dataset
    
    # Charger et prétraiter les données
    text = load_dataset('../data/dataset.txt')
    preprocessor = TextPreprocessor(max_vocab_size=100)
    cleaned_text = preprocessor.clean_text(text)
    preprocessor.build_vocabulary(cleaned_text)
    
    # Convertir le texte en séquences
    sequences = preprocessor.text_to_sequences(cleaned_text)
    
    # Créer et entraîner le modèle
    generator = TextGeneratorLSTM(vocab_size=len(preprocessor.vocabulary),
                                sequence_length=5)
    X, y = generator.prepare_sequences([sequences])
    
    print("Entraînement du modèle LSTM...")
    generator.train(X, y, epochs=20)  # Époques réduites pour démonstration
    
    # Générer du texte
    seed = sequences[:5]  # Utiliser les 5 premiers mots comme graine
    generated_indices = generator.generate(seed)
    
    # Convertir les indices en mots
    generated_words = [preprocessor.idx2word.get(idx, '<UNK>') 
                      for idx in generated_indices]
    print("\nTexte généré :")
    print(' '.join(generated_words))

if __name__ == "__main__":
    main()
