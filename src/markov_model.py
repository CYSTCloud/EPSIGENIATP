import random
from collections import defaultdict
from typing import List, Dict, Tuple

class MarkovChain:
    def __init__(self, order: int = 1):
        """
        Initialiser la chaîne de Markov.
        Args:
            order (int): L'ordre de la chaîne de Markov (nombre de mots à considérer comme état)
        """
        self.order = order
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.starts = []

    def _get_ngrams(self, tokens: List[str]) -> List[Tuple[tuple, str]]:
        """Générer les n-grammes à partir du texte."""
        ngrams = []
        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i:i + self.order])
            next_word = tokens[i + self.order]
            ngrams.append((state, next_word))
        return ngrams

    def train(self, text: str):
        """
        Entraîner la chaîne de Markov sur le texte d'entrée.
        Args:
            text (str): Texte d'entrée pour l'entraînement
        """
        tokens = text.lower().split()
        
        for i in range(len(tokens) - self.order):
            if i == 0 or tokens[i-1].endswith('.'):
                self.starts.append(tuple(tokens[i:i + self.order]))

        if not self.starts:
            self.starts.append(tuple(tokens[:self.order]))

        for state, next_word in self._get_ngrams(tokens):
            self.transitions[state][next_word] += 1

    def _choose_next(self, state: tuple) -> str:
        """Choisir le prochain mot en fonction de l'état actuel."""
        if state not in self.transitions:
            return None
        
        choices = list(self.transitions[state].items())
        words, weights = zip(*choices)
        return random.choices(words, weights=weights)[0]

    def generate(self, max_words: int = 20) -> str:
        """
        Générer du texte en utilisant le modèle entraîné.
        Args:
            max_words (int): Nombre maximum de mots à générer
        Returns:
            str: Texte généré
        """
        if not self.starts:
            return ""
            
        current = random.choice(self.starts)
        result = list(current)

        for _ in range(max_words - self.order):
            next_word = self._choose_next(tuple(current))
            if next_word is None:
                break
            
            result.append(next_word)
            current = tuple(result[-self.order:])

        return ' '.join(result)

def main():
    with open('../data/dataset.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    model = MarkovChain(order=2)
    model.train(text)
    generated_text = model.generate(max_words=50)
    print(generated_text)

if __name__ == "__main__":
    main()
