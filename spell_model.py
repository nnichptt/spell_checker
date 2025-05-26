# spell_model.py
import math
import nltk
from nltk.corpus import words, brown
from nltk.metrics import edit_distance
from collections import Counter

class SpellChecker:
    def __init__(self):
       # Load dictionary from nltk corpus
        try:
            nltk.data.find('corpora/words')
            nltk.data.find('corpora/brown')
        except:
            print('Corpora not Found start downloading...')
            nltk.download('words')
            nltk.download('brown')

        self.dictionary = set(words.words())
        self.freqs = Counter([w.lower() for w in brown.words()])
        # filter rare junk
        self.dictionary = {w for w, c in self.freqs.items() if c >= 3}

    def suggest(self, word: str, max_candidates: int = 5) -> list[str]:
        """
        Return up to max_candidates suggestions ordered by increasing distance.
        """
        word = word.lower()
        # Compute distance to every dictionary word (inefficient for large dict)
        # For real use, integrate BK-tree for pruning.
        dists = []
        for w in self.dictionary:
            d = edit_distance(word, w, transpositions=True)
            dists.append((d, w))
        # sort by distance then alphabetically
        dists.sort(key=lambda x: (x[0], x[1]))
        # return only words at minimal distances
        suggestions = [w for d, w in dists if d <=
                       dists[0][0]][:max_candidates]
        return suggestions

    def brownSuggest(self, word: str, max_candidates: int = 5) -> list[str]:
        """
        Return up to max_candidates suggestions ordered by increasing distance. Using Brown dataset.
        """
        word = word.lower()

        # Step 1: Use BK-tree to get candidates within edit distance 2
        candidates = self.dictionary

        # Step 2: Score candidates by (edit distance, -log(freq))
        def score(w):
            dist = edit_distance(word, w, transpositions=True)
            # fallback to log(1) = 0
            freq_score = -math.log(self.freqs.get(w, 1))
            return (dist, freq_score)

        # Step 3: Sort candidates and pick top N
        ranked = sorted(candidates, key=score)
        return ranked[:max_candidates]
