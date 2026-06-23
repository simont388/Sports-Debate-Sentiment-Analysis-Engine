import re
import os
import json

# Combine stat paragraphs, Wikipedia text, and articles into a single corpus.json
# Each chunk gets source, player, debate tags, and argument_side metadata

STAT_PARAGRAPHS_FILE = "data/corpus/stat_paragraphs.json"
WIKIPEDIA_DIR = "data/raw/wikipedia"
ARTICLES_DIR = "data/raw/articles"
OUTPUT_FILE = "data/corpus/corpus.json"

# Higher weight for more reliable sources
SOURCE_WEIGHTS = {
    "basketball_reference": 1.0,
    "wikipedia": 0.8,
    "articles": 0.6
}

MIN_PARAGRAPH_LENGTH = 75

# Map Wikipedia filenames to player names and associated debate IDs
WIKIPEDIA_FILE_MAP = {
     "lebron_james":           {
        "player":  "LeBron James",
        "debates": ["lebron_washed", "kobe_vs_lebron_scoring"]
    },
    "stephen_curry":          {
        "player":  "Stephen Curry",
        "debates": ["curry_system_player", "warriors_2017_goat_team"]
    },
    "giannis_antetokounmpo":  {
        "player":  "Giannis Antetokounmpo",
        "debates": ["giannis_no_costar"]
    },
    "kobe_bryant":            {
        "player":  "Kobe Bryant",
        "debates": ["kobe_vs_lebron_scoring"]
    },
    "kawhi_leonard":          {
        "player":  "Kawhi Leonard",
        "debates": ["kawhi_overrated"]
    },
    "russell_westbrook":      {
        "player":  "Russell Westbrook",
        "debates": ["westbrook_ruined_lakers"]
    },
    "carmelo_anthony":        {
        "player":  "Carmelo Anthony",
        "debates": ["carmelo_selfish"]
    },
    "draymond_green":         {
        "player":  "Draymond Green",
        "debates": ["draymond_most_important_warrior"]
    },
    "nikola_jokic":           {
        "player":  "Nikola Jokic",
        "debates": ["jokic_most_complete"]
    },
    "warriors_2017":          {
        "player":  "Golden State Warriors",
        "debates": ["warriors_2017_goat_team"]
    },
}

DEBATE_KEYWORD_MAP = {
    "lebron_washed" : ["lebron_washed", "kobe_vs_lebron_scoring"],
    "steph" : ["curry_system_player", "warriors_2017_goat_team"],
    "giannis" : ["giannis_no_costar"],
    "lebron_kobe" : ["kobe_vs_lebron_scoring"],
    "carmelo" : ["carmelo_selfish"],
    "draymond" : ["draymond_most_important_warrior", "warriors_2017_goat_team"],
    "jokic" : ["jokic_most_complete"],
    "kawhi" : ["kawhi_overrated"],
    "warriors" : ["warriors_2017_goat_team"],
    "westbrook" : ["westbrook_ruined_lakers"]
}

# Infer article stance from filename keywords (e.g. "giannis_espn_support.txt")
STANCE_KEYWORD_MAP = {
    "support" : "supporting",
    "against" : "opposing",
    "neutral" : "neutral"
}

def is_reference_paragraph(text):
    # Detect bibliography, citations, or reference section lines
    text_stripped = text.strip()
    text_lower = text_stripped.lower()

    section_headers = {
        "see also", "notes", "references", "further reading",
        "external links", "bibliography", "sources", "works cited"
    }
    if text_lower.rstrip("s") in section_headers or text_lower in section_headers:
        return True

    if re.search(r'\bISBN\b', text_stripped):
        return True

    if re.match(r'^[A-Z][a-z]+,\s*[A-Z]\.?\s*[A-Z]?\.?\s*\(', text_stripped):
        return True

    if text_lower.startswith("career statistics from"):
        return True

    if text_lower.startswith("official website"):
        return True

    return False

def is_noise_paragraph(text):
    # Detect Wikipedia artifacts: charity mentions, birth info, school history
    text_lower = text.lower()
    
    # Regex patterns for structured noise
    regex_patterns = [
        r'donated\s+\$\s*\d+(?:,\d{3})*',
        r'pledged?\s+.*?\$\s*\d+(?:,\d{3})*',
        r'committed?\s+\$?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand)?\s*(?:toward|to\s+(?:help|fund|build|support|provide))',
        r'\bbecame\s+a\s+shareholder\b',
        r'\b(?:charit(?:y|able)|philanthrop(?:y|ist|ic)|non.profit|nonprofit)\b',
        r'isbn\s*[\d\-x]+',  # catches ISBN references
    ]
    
    # Simple substring patterns for Wikipedia artifacts
    substring_patterns = [
        "archived, alternate link",
        "alternate link",
        "wristband",
        "why not? foundation",
        "launched a weekly podcast",
        "born in",
        "car accident",
        "high school",
        "elementary school",
        "middle school",
        "signed a one-year",
        "veterans minimum",
        "summer olympics",
    ]
    
    # Check regex patterns
    if any(re.search(p, text_lower) for p in regex_patterns):
        return True
    
    # Check substring patterns
    if any(p in text_lower for p in substring_patterns):
        return True
    
    # Filter very short chunks — single sentence artifacts
    # Legitimate analytical paragraphs are almost always multi-sentence
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) < 2:
        return True
    
    return False


def clean_text(text):
    # Remove short lines without punctuation (likely headings or artifacts)
    lines = [line.strip() for line in text.splitlines()]
    cleaned = []
    for line in lines:
        if not line:
            continue
        if len(line) < 30 and not any(c in line for c in ".,:;!?"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def split_into_paragraphs(text):
    # Split on double newlines; fall back to single newlines if too few chunks
    chunks = re.split(r'\n{2,}', text)
    if len(chunks) < 3:
        chunks = text.split('\n')

    paragraphs = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > MIN_PARAGRAPH_LENGTH:
            paragraphs.append(chunk)
    return paragraphs

def get_debate_from_file(filename):
    # Map article filename to debate IDs using keyword matching
    filename_lower = filename.lower()
    for keyword, debates in DEBATE_KEYWORD_MAP.items():
        if keyword in filename_lower:
            return debates
    return ["unknown debate"]

def get_stance_from_file(filename):
    # Infer article stance (supporting/opposing) from filename
    filename_lower = filename.lower()
    for keyword, stance in STANCE_KEYWORD_MAP.items():
        if keyword in filename_lower:
            return stance
    return "neutral"

def load_stat_paragraphs():
    if not os.path.exists(STAT_PARAGRAPHS_FILE):
        print(f"Error: {STAT_PARAGRAPHS_FILE} not found. Please run stats_template.py first.")        
        return []
    with open(STAT_PARAGRAPHS_FILE, "r", encoding="utf-8") as f:
        paragraphs = json.load(f)
    
    print(f"Loaded {len(paragraphs)} stat paragraphs from {STAT_PARAGRAPHS_FILE}")
    return paragraphs

def load_wikipedia_paragraphs():
    if not os.path.exists(WIKIPEDIA_DIR):
        print(f"Error: {WIKIPEDIA_DIR} not found. Please run wikipedia_crawler.py first.")
        return []
    all_paragraphs = []
    for filename in os.listdir(WIKIPEDIA_DIR):
        if not filename.endswith(".txt"):
            continue
        file_key = filename.replace("_wiki.txt", "").lower()
        meta = WIKIPEDIA_FILE_MAP.get(file_key)
        if not meta:
            print(f"Warning: No metadata found for {filename}, skipping.")
            continue
        filepath = os.path.join(WIKIPEDIA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
        cleaned = clean_text(raw_text)
        raw_paragraphs = split_into_paragraphs(cleaned)
        # Filter out references, citations, and noise (birth info, charity, etc.)
        paragraphs = [p for p in raw_paragraphs if not is_reference_paragraph(p) and not is_noise_paragraph(p)]
        for i, chunk in enumerate(paragraphs):
            all_paragraphs.append({
                "text": chunk,
                "source": f"Wikipedia - {meta['player']}",
                "source_weight": SOURCE_WEIGHTS["wikipedia"],
                "player": meta["player"],
                "season": "career",
                "paragraph_type": "wikipedia",
                "chunk_index": i,
                "debates": meta["debates"],
                "argument_side": "neutral"
            })
        
        print(f"Loaded Wikipedia: {filename} -> {len(paragraphs)} paragraphs")
    
    print(f"Total Wikipedia paragraphs: {len(all_paragraphs)}")
    return all_paragraphs

def load_article_paragraphs():
    # Parse articles with 3-line header (title, author, source) + body text
    if not os.path.exists(ARTICLES_DIR):
        print(f"Error: {ARTICLES_DIR} not found.")
        return []
    all_paragraphs = []
    for filename in sorted(os.listdir(ARTICLES_DIR)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(ARTICLES_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 4:
            print(f"WARNING: {filename} is too short")
            continue

        title = lines[0].strip()
        author = lines[1].strip()
        source = lines[2].strip()

        if not title:
            print(f"WARNING: {filename} has no title")
        if not author:
            print(f"WARNING: {filename} has no author")
        if not source:
            print(f"WARNING: {filename} has no source")
        raw_text = "".join(lines[3:])
        cleaned = clean_text(raw_text)
        chunks = split_into_paragraphs(cleaned)
        if not chunks:
            print(f"WARNING: {filename} produced no paragraphs after filtering")
            continue

        debates = get_debate_from_file(filename)
        side = get_stance_from_file(filename)
        if "unknown debate" in debates:
            print(f"WARNING: could not get debate for {filename}")
        
        for i, chunk in enumerate(chunks):
            all_paragraphs.append({
                "text":           chunk,
                "source":         source,
                "author":         author,
                "title":          title,
                "source_weight":  SOURCE_WEIGHTS["articles"],
                "player":         "various",
                "season":         "various",
                "paragraph_type": "article",
                "chunk_index":    i,
                "filename":       filename,
                "debates":        debates,
                "argument_side":  side
            })
        
        print(f"Loaded: {filename}\nTitle: {title}\nStance: {side}\nSource: {source}\nChunks: {len(chunks)}")
    print(f"\nTotal Article Paragraphs: {len(all_paragraphs)}")
    return all_paragraphs

def build_corpus():
    print("=== Building Full Corpus ===")
    all_paragraphs = []
    print("--- Loading Stat Paragraphs ---")
    stat_paragraphs = load_stat_paragraphs()
    all_paragraphs.extend(stat_paragraphs)
    print("--- Loading Wikipedia Paragraphs ---")
    wiki_paragraphs = load_wikipedia_paragraphs()
    all_paragraphs.extend(wiki_paragraphs)
    print("--- Loading Article Paragraphs ---")
    article_paragraphs = load_article_paragraphs()
    all_paragraphs.extend(article_paragraphs)

    # Assign unique chunk IDs for LSI indexing
    for i, para in enumerate(all_paragraphs):
        para["chunk_id"] = i
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf=8") as f:
        json.dump(all_paragraphs, f, indent=2, ensure_ascii=False)
    
    print(f"=== CORPUS SUMMARY ===")
    print(f"Total Chunks: {len(all_paragraphs)}")
    print(f"Stat Paragraphs: {len(stat_paragraphs)}")
    print(f"Wikipedia Paragraphs: {len(wiki_paragraphs)}")
    print(f"Article Paragraphs: {len(article_paragraphs)}")

    debate_counts = {}
    for para in all_paragraphs:
        for debate in para.get("debates", []):
            debate_counts[debate] = debate_counts.get(debate, 0) + 1
    print(f"\n--- Chunks Per Debate ---")
    for debate, count in sorted(debate_counts.items()):
        print(f"{debate}: {count}")
    
    side_counts = {}
    for para in all_paragraphs:
        side = para.get("argument_side", "neutral")
        side_counts[side] = side_counts.get(side, 0) + 1
    print(f"\n--- Chunks By Argument Side ---")
    for side, count in sorted(side_counts.items()):
        print(f"{side}: {count}")


if __name__ == "__main__":
    build_corpus()

        
        


