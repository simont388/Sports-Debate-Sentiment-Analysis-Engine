import wikipediaapi

# Download Wikipedia article text for each player/team
# Output saved to data/raw/wikipedia/{name}_wiki.txt

wiki = wikipediaapi.Wikipedia(user_agent='sports_debate-research/1.0', language='en')
pages = {
    # Players
    "lebron_james":           "LeBron James",
    "stephen_curry":          "Stephen Curry",
    "giannis_antetokounmpo":  "Giannis Antetokounmpo",
    "kobe_bryant":            "Kobe Bryant",
    "kawhi_leonard":          "Kawhi Leonard",
    "russell_westbrook":      "Russell Westbrook",
    "carmelo_anthony":        "Carmelo Anthony",
    "draymond_green":         "Draymond Green",
    "nikola_jokic":           "Nikola Jokic",
    # Teams
    "warriors_2017":          "2016-17 Golden State Warriors season",
}

for filename, page_title in pages.items():
    page = wiki.page(page_title)
    if not page.exists():
        print(f"Page not found: {page_title}")
        continue
    filepath = f"data/raw/wikipedia/{filename}_wiki.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(page.text)
    print(f"Saved: {filepath}")