import json
import urllib.request
import urllib.error

# Stage 6 — Generate a natural-language verdict using Ollama LLM (with rule-based fallback)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
TOP_N_CONTEXT = 3
MAX_TOKENS = 200

def call_ollama(prompt, model=OLLAMA_MODEL):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": 0.3
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"  Ollama not available ({e}). Using rule-based verdict fallback.")
        return None

def rule_based_verdict(claim):
    # Fallback verdict when Ollama is unavailable, based purely on scoring signals
    verdict_lean = claim.get("verdict_lean", "mixed evidence")
    raw_query = claim.get("raw_query", "")

    if verdict_lean == "supports the claim":
        summary = (
            f"The evidence largely supports the assertion that {raw_query.lower()}. "
            f"Statistical data and analytical sources show consistent patterns "
            f"aligned with this claim, though some counter-evidence exists."
        )
    elif verdict_lean == "refutes the claim":
        summary = (
            f"The evidence largely refutes the assertion that {raw_query.lower()}. "
            f"Statistical data and analytical sources suggest the opposite conclusion, "
            f"though the debate has merit on both sides."
        )
    else:
        summary = (
            f"The evidence on '{raw_query}' is genuinely mixed. "
            f"Statistical data and analytical sources present compelling arguments "
            f"on both sides, making this a legitimately contested debate."
        )
    
    return summary

def build_verdict_prompt(claim, top_a, top_b):
    raw_query = claim.get("raw_query", "")
    confidence = claim.get("confidence_score", 0.5)
    verdict_lean = claim.get("verdict_lean", "mixed evidence")

    supporting_text = ""
    for i, chunk in enumerate(top_a):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")[:300]
        supporting_text += f"\n [{i}] ({source}): {text}"
    
    refuting_text = ""
    for i, chunk in enumerate(top_b):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")[:300]
        refuting_text += f"\n [{i}] ({source}): {text}"

    # Force the LLM to follow the scoring-based verdict lean
    if verdict_lean == "supports the claim":
        conclusion_instruction = f"Your verdict must conclude that: {raw_query}"
    elif verdict_lean == "refutes the claim":
        conclusion_instruction = f"Your verdict must conclude that the following assertion is FALSE: {raw_query}"
    else:
        conclusion_instruction = f"Your verdict must conclude that the evidence on this debate is genuinely split and no conclusion can be made either way: {raw_query}"

    prompt = f"""You are a bold sports analytics expert who always takes a clear stance.

            DEBATE: "{raw_query}"

            EVIDENCE SUPPORTING THE CLAIM:
            {supporting_text if supporting_text else "No strong supporting evidence found."}

            EVIDENCE REFUTING THE CLAIM:
            {refuting_text if refuting_text else "No strong refuting evidence found."}

            {conclusion_instruction}

            Instructions:
            - Write exactly two sentences
            - State your conclusion immediately in the first sentence
            - Cite specific statistics or facts from the evidence in the second sentence
            - Do not start with "I", "Based on", or "The evidence"
            - Do not hedge
            - Do not provide counterpoints unless the verdict is mixed evidence
            - Write as a confident sports analyst taking a definitive position
            - If the evidence is genuinely split then say so and point to both sides of the debate
            - If the evidence is genuinely split, take a neutral stance and explain both sides
            - if the evidence is split do not provide a conclusion leaning one way

            VERDICT:"""
    
    return prompt

def select_top_chunks(scored_chunks, n=TOP_N_CONTEXT):
    # Prioritize high-confidence chunks first, then medium, then low
    high = [c for c in scored_chunks if c.get("side_confidence") == "high"]
    medium = [c for c in scored_chunks if c.get("side_confidence") == "medium"]
    low = [c for c in scored_chunks if c.get("side_confidence") == "low"]

    high.sort(key=lambda x: x.get("final_score",0), reverse=True)
    medium.sort(key=lambda x: x.get("final_score",0), reverse=True)
    low.sort(key=lambda x: x.get("final_score",0), reverse=True)

    selected = []
    for tier in [high, medium, low]:
        for chunk in tier:
            if len(selected) >= n:
                break
            selected.append(chunk)
        if len(selected) >= n:
            break
    
    return selected

def generate_verdict(claim):
    scored_a = claim.get("scored_a", [])
    scored_b = claim.get("scored_b", [])

    top_a = select_top_chunks(scored_a, TOP_N_CONTEXT)
    top_b = select_top_chunks(scored_b, TOP_N_CONTEXT)

    prompt = build_verdict_prompt(claim, top_a, top_b)
    verdict_text = call_ollama(prompt)
    method = "ollama"

    if not verdict_text:
        verdict_text = rule_based_verdict(claim)
        method = "rule_based"
    
    claim["verdict"] = {
        "verdict_text": verdict_text,
        "verdict_lean": claim.get("verdict_lean", "mixed_evidence"),
        "confidence": round(claim.get("confidence_score", 0.5), 3),
        "top_supporting": top_a,
        "top_refuting": top_b,
        "method": method
    }

    return claim

def test_verdict():
    from query_parser import parse_query, load_spacy
    from sentiment import SentimentAnalyzer
    from query_expansion import expand_query, load_lexicon, load_refutation_map
    from retrieval import retrieve, load_paragraph_model, load_stopwords
    from scoring import score_results
 
    stopwords = load_stopwords()
    model = load_paragraph_model()
    lexicon = load_lexicon()
    refutation_map = load_refutation_map()
    nlp = load_spacy()
    analyzer = SentimentAnalyzer()

    test_queries = [
        "LeBron is washed",
        "Kawhi Leonard is overrated",
        "Kobe was a better scorer than LeBron",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
 
        claim = parse_query(query, nlp=nlp, lexicon=lexicon)
        claim = analyzer.enrich_claim(claim)
        claim = expand_query(claim, lexicon=lexicon, refutation_map=refutation_map)
        claim = retrieve(claim, model=model, stopwords=stopwords)
        claim = score_results(claim)
        claim = generate_verdict(claim)
 
        verdict = claim["verdict"]
 
        print(f"\n  VERDICT ({verdict['verdict_lean'].upper()}, "
              f"confidence={verdict['confidence']:.3f}, "
              f"method={verdict['method']})")
        print(f"\n  {verdict['verdict_text']}")
 
        print(f"\n  TOP SUPPORTING EVIDENCE:")
        for i, chunk in enumerate(verdict["top_supporting"], 1):
            print(f"    {i}. [{chunk['source']}] "
                  f"score={chunk['final_score']:.3f} "
                  f"conf={chunk['side_confidence']}")
            print(f"       {chunk['text'][:150]}...")
 
        print(f"\n  TOP REFUTING EVIDENCE:")
        for i, chunk in enumerate(verdict["top_refuting"], 1):
            print(f"    {i}. [{chunk['source']}] "
                  f"score={chunk['final_score']:.3f} "
                  f"conf={chunk['side_confidence']}")
            print(f"       {chunk['text'][:150]}...")
 

if __name__ == "__main__":
    test_verdict()