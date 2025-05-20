# spell_model.py

from collections import defaultdict
import math
import nltk
from nltk.corpus import words
dictionary = set(words.words())


def min_edit_distance(word1, word2):
    len1, len2 = len(word1), len(word2)
    
    # Create a distance matrix
    dp = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]

    # Initialize first row and column
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[len1][len2]

# --- Weighted MED / Damerau‑Levenshtein (optional) ---
def damerau_levenshtein(w1, w2):
    # ... implement or import a library version ...
    return min_edit_distance(w1, w2)  # placeholder

# --- Load frequencies from a simple CSV “word,frequency” file ---
def load_frequency(file_path="word_freq.csv"):
    freqs = {}
    with open(file_path) as f:
        for line in f:
            word, count = line.strip().split(',')
            freqs[word] = int(count)
    # Convert to log‑probabilities
    total = sum(freqs.values())
    return {w: math.log(c/total) for w,c in freqs.items()}

# --- BK‑Tree construction for fast “within k edits” search ---
class BKTree:
    def __init__(self, dist_fn):
        self.dist_fn = dist_fn
        self.tree = None

    def add(self, term):
        node = (term, {})
        if self.tree is None:
            self.tree = node
        else:
            self._add(self.tree, node)

    def _add(self, node, new_node):
        term, children = node
        new_term, _ = new_node
        d = self.dist_fn(new_term, term)
        if d in children:
            self._add(children[d], new_node)
        else:
            children[d] = new_node

    def query(self, term, max_dist):
        return [n for d in range(max_dist+1)
                   for n in self._search(self.tree, term, d)]

    def _search(self, node, term, dist):
        if node is None: return []
        term0, children = node
        d0 = self.dist_fn(term, term0)
        results = [term0] if d0 == dist else []
        for edge_dist, child in children.items():
            # only descend into plausible subtrees
            if abs(edge_dist - dist) <= dist:
                results += self._search(child, term, dist)
        return results

# --- Putting it together ---
class SpellChecker:
    def __init__(self):
       # Load dictionary from nltk corpus
        try:
            print(nltk.data.find('/Users/nichapatpeasri/nltk_data/corpora/words.zip'))
        except:
            self.dictionary = nltk.download('words')
        
        self.dictionary = set(words.words())

    def suggest(self,word: str, max_candidates: int = 5) -> list[str]:
        """
        Return up to max_candidates suggestions ordered by increasing distance.
        """
        word = word.lower()
        # Compute distance to every dictionary word (inefficient for large dict)
        # For real use, integrate BK-tree for pruning.
        dists = []
        for w in self.dictionary:
            d = damerau_levenshtein(word, w)
            dists.append((d, w))
        # sort by distance then alphabetically
        dists.sort(key=lambda x: (x[0], x[1]))
        # return only words at minimal distances
        suggestions = [w for d, w in dists if d <= dists[0][0]][:max_candidates]
        return suggestions

