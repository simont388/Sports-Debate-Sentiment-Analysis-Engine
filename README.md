Claims:

LeBron is washed.
Steph Curry is a system player.
Giannis can't win a championship without a co-star
Kobe was a better scorer than LeBron.
The 2017 Warriors are the greatest team of all time
Kawhi Leonard is overrated
Westbrook ruined the Lakers
Carmelo Anthony was never a winner because he was selfish
Draymond Green is the most important player on the Warriors
Nikola Jokic is the most complete player of all time

Usage:

Preprocessing:

python main.py "LeBron is washed"            # single debate
python main.py --all                          # run all 10 fixed debates
python main.py --interactive                  # interactive REPL mode

Requires `spacy`, `gensim`, `nltk`, `pandas`, `scikit-learn`, `numpy`. LSI models and GloVe vectors load on startup. Optional: Ollama with `llama3.2` for LLM verdicts (falls back to rule-based).

Architecture:

sports_debate_ir/
│
├── data/
│   ├── raw/
│   │   ├── articles/                   # ~35 editorial articles with stance labels
│   │   └── wikipedia/                  # 10 player/team Wikipedia pages
│   ├── stats/                          # per-game + advanced CSVs from Basketball Reference
│   └── corpus/
│       ├── corpus.json                 # all paragraphs merged with metadata
│       └── stat_paragraphs.json        # synthetic stat paragraphs pre-built
│
├── config/
│   ├── word_mapping.json               # slang terms → concepts, stat dims, claim types
│   ├── refutation.json                 # counter-argument concepts per claim type
│   └── common_words                    # stopword list for TF-IDF
│
├── preprocessing/                   
│   ├── bbref_crawler.py                # scrapes Basketball Reference season + advanced stats
│   ├── wikipedia_crawler.py            # downloads Wikipedia article text
│   ├── stats_template.py               # CSV rows → synthetic stat paragraphs
│   ├── build_corpus.py                 # merges stats + Wikipedia + articles → corpus.json
│   └── build_lsi.py                    # TF-IDF → TruncatedSVD → paragraph + doc LSI models
│
├── query_parser.py                     # [Stage 1] spaCy NER, slang detection, claim classification
├── sentiment.py                        # [Stage 2] GloVe + WordNet concept expansion
├── query_expansion.py                  # [Stage 3] supporting (Set A) / refuting (Set B) query sets
├── retrieval.py                        # [Stage 4] LSI search + stat trend/comparison lookups
├── scoring.py                          # [Stage 5] contrastive scoring: 0.5·LSI + 0.3·source + 0.2·specificity
├── verdict.py                          # [Stage 6] Ollama LLM verdict (fallback: rule-based)
│
├── models/
│   ├── lsi_paragraphs.pkl              # paragraph-level LSI (k=150)
│   └── lsi_documents.pkl               # document-level LSI (k=50) for debate filtering
│
├── results/
│   └── all_debate_results.txt          # batch output for --all mode
│
└── main.py                             # entry point, 6-stage pipeline orchestrator
