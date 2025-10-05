import json
import re

# Read the TSV file
input_file = "/home/yukaalive/2025workspace/task_vectors/21_icl_task_vectors/raw/short_200.tsv"
output_file = "/home/yukaalive/2025workspace/task_vectors/21_icl_task_vectors/data/translation/ja_en.json"

def clean_text(text):
    """Remove special characters and clean text"""
    # Remove brackets and their content
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'《.*?》', '', text)
    text = re.sub(r'「.*?」', '', text)
    # Remove special markers
    text = re.sub(r'[♪\u266a\u266b]', '', text)
    # Remove leading/trailing punctuation
    text = text.strip('!?.,:;…。、！？：；')
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_tokens_approx(text):
    """Approximate token count (words for English, characters for Japanese)"""
    # For English: count words
    if re.search(r'[a-zA-Z]', text):
        return len(text.split())
    # For Japanese: count characters (rough approximation)
    else:
        return len(text)

def is_appropriate_length(english, japanese, min_tokens=2, max_tokens=15):
    """Check if both texts have appropriate length (multiple tokens)"""
    en_tokens = count_tokens_approx(english)
    ja_tokens = count_tokens_approx(japanese)

    # Both should have at least min_tokens to ensure multiple tokens
    # and at most max_tokens to keep them reasonably short
    return (min_tokens <= en_tokens <= max_tokens and
            min_tokens <= ja_tokens <= max_tokens)

# Read and parse the TSV file
translations = {}
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Split by tab
        parts = line.split('\t')
        if len(parts) != 2:
            continue

        english, japanese = parts[0].strip(), parts[1].strip()

        # Clean texts
        english = clean_text(english)
        japanese = clean_text(japanese)

        # Skip if empty after cleaning
        if not english or not japanese:
            continue

        # Skip if not appropriate length (want multiple tokens)
        if not is_appropriate_length(english, japanese):
            continue

        # Convert to lowercase for English
        english_lower = english.lower()

        # Add to dictionary (reverse: Japanese -> English)
        translations[japanese] = english_lower

        # Stop when we have 200 pairs
        if len(translations) >= 200:
            break

# Save to JSON file with proper formatting
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(translations, f, ensure_ascii=False, indent=4)

print(f"Created {output_file} with {len(translations)} translation pairs")
print(f"Sample entries:")
for i, (k, v) in enumerate(list(translations.items())[:5]):
    print(f"  {i+1}. '{k}' -> '{v}'")
