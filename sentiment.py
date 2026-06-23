import nltk
from nltk.corpus import wordnet
import gensim.downloader as api

# Stage 2 — Sentiment enrichment via GloVe vectors and WordNet synonyms

try:
    wordnet.ensure_loaded()
except:
    print(f"Downloading wordnet")
    nltk.download('wordnet')

class SentimentAnalyzer:
    def __init__(self, model_name="glove-wiki-gigaword-100"):
        print(f"Loading {model_name} vectors into memory")
        self.glove = api.load(model_name)
        # Seed words that are always considered basketball-relevant
        self.domain_filter = {
            'player', 'season', 'career', 'stats', 'game', 'performance', 
            'scoring', 'play', 'team', 'league', 'history', 'year', 'age',
            'athletic', 'prime', 'physical', 'minutes', 'points', 'rebounds',
            'efficiency', 'all-star', 'mvp', 'playoffs', 'championship'
        }

    def _is_relevant(self, word):
        # Check if a word is basketball-relevant using domain filter or GloVe similarity
        word = word.lower()
        if word in self.domain_filter:
            return True

        try:
            score_athlete = self.glove.similarity(word, 'athlete')
            score_bball = self.glove.similarity(word, 'basketball')
            return score_athlete > 0.25 or score_bball > 0.25
        except KeyError:
            return False

    def _get_wordnet_synonyms(self, word):
        # Find synonyms via WordNet, filtered to basketball relevance
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                clean_lemma = lemma.name().replace('_', ' ').lower()
                if self._is_relevant(clean_lemma):
                    synonyms.add(clean_lemma)
        return synonyms
    
    def _get_glove_neighbors(self, word, top_n=20):
        # Find semantically similar words via GloVe cosine similarity > 0.6
        neighbors = set()
        try:
            results = self.glove.most_similar(word, topn=top_n)
            for neighbor, score in results:
                neighbor = neighbor.lower()
                if score > 0.6 and self._is_relevant(neighbor):
                    neighbors.add(neighbor.lower())
        except KeyError:
            pass
        
        return neighbors

    def enrich_claim(self, claim):
        # Expand claim concepts using WordNet synonyms and GloVe neighbors
        base_concepts = claim.get("concepts", [])
        slang_terms = claim.get("slang_terms", [])

        seeds = list(set(base_concepts + slang_terms))
        expanded_pool = set(seeds)

        for seed in seeds:
            expanded_pool.update(self._get_wordnet_synonyms(seed))
            expanded_pool.update(self._get_glove_neighbors(seed))
        
        # Remove entity names (don't want player names in concept expansion)
        entities = [e.lower() for e in claim.get("entities", [])]
        final_concepts = [
            word for word in expanded_pool if word not in entities and len(word) > 2
        ]

        claim["expanded_concepts"] = sorted(list(set(final_concepts)))

        return claim
    
if __name__ == "__main__":
    test_claim = {
        "raw_query": "LeBron is washed",
        "entities": ["LeBron James"],
        "slang_terms": ["washed"],
        "concepts": ["decline", "aging", "regression"], # From word_mapping.json
        "stat_dimensions": ["PER", "PPG"]
    }

    analyzer = SentimentAnalyzer()
    enriched = analyzer.enrich_claim(test_claim)
    print("\n" + "="*30)
    print(f"QUERY: {enriched['raw_query']}")
    print(f"EXPANDED CONCEPTS: {enriched['expanded_concepts']}")
    print("="*30)