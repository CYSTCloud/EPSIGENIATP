import re
from collections import Counter
from typing import List, Dict

class TextPreprocessor:
    def __init__(self, max_vocab_size: int = 1000):
        self.max_vocab_size = max_vocab_size
        self.vocab = None
        self.word_to_index = None
        self.index_to_word = None

    def clean_text(self, text: str) -> str:
        """Nettoyer et normaliser le texte."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def build_vocabulary(self, text: str) -> Dict[str, int]:
        """Construire le vocabulaire à partir du texte avec une taille limitée."""
        words = text.split()
        word_counts = Counter(words)
        most_common = word_counts.most_common(self.max_vocab_size)
        self.vocab = {word: count for word, count in most_common}
        self.word_to_index = {word: i for i, (word, _) in enumerate(most_common)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        return self.vocab

    def text_to_sequences(self, text: str) -> List[int]:
        """Convertir le texte en séquence d'indices."""
        words = text.split()
        return [self.word_to_index.get(word, 0) for word in words]

    def sequences_to_text(self, sequences: List[int]) -> str:
        """Convertir les séquences en texte."""
        return ' '.join([self.index_to_word.get(idx, '') for idx in sequences])

def load_dataset(file_path: str) -> str:
    """Charger le jeu de données texte à partir d'un fichier."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    text = load_dataset('../data/dataset.txt')
    preprocessor = TextPreprocessor(max_vocab_size=100)
    cleaned_text = preprocessor.clean_text(text)
    vocab = preprocessor.build_vocabulary(cleaned_text)
    sequences = preprocessor.text_to_sequences(cleaned_text)
    reconstructed_text = preprocessor.sequences_to_text(sequences)

if __name__ == '__main__':
    main()
