#!/usr/bin/env python3
"""
Test script for COMET and BLEU evaluation in TranslationTask
"""

import sys
sys.path.append('.')

from transformers import AutoTokenizer
from core.data.tasks.translation_task import TranslationTask
from typing import List, Dict


def test_translation_evaluation():
    """Test the new COMET and BLEU evaluation methods"""
    
    # Initialize tokenizer (using a simple one for testing)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test data for different language pairs
    test_cases = {
        "en_fr": {
            "sources": [
                "Hello world",
                "How are you?",
                "This is a test",
                "Good morning",
                "Thank you very much"
            ],
            "predictions": [
                "Salut le monde",
                "Comment allez-vous?",
                "Ceci est un test",
                "Bon matin",
                "Merci beaucoup"
            ],
            "references": [
                "Bonjour le monde",
                "Comment ça va?",
                "C'est un test",
                "Bonjour",
                "Merci beaucoup"
            ]
        },
        "en_es": {
            "sources": [
                "Hello world",
                "How are you?",
                "This is a test",
                "Good morning",
                "Thank you very much"
            ],
            "predictions": [
                "Hola mundo",
                "¿Cómo estás?",
                "Esto es una prueba",
                "Buenos días",
                "Muchas gracias"
            ],
            "references": [
                "Hola mundo",
                "¿Cómo está usted?",
                "Esta es una prueba",
                "Buenos días",
                "Muchas gracias"
            ]
        },
        "en_ja": {
            "sources": [
                "Hello world",
                "How are you?",
                "This is a test",
                "Good morning", 
                "Thank you very much"
            ],
            "predictions": [
                "こんにちは世界",
                "元気ですか？",
                "これはテストでござるのまき",
                "おはよう",
                "どうもありがとう"
            ],
            "references": [
                "こんにちは、世界",
                "お元気ですか？",
                "これはテストです",
                "おはようございます",
                "ありがとうございます"
            ]
        }
    }
    
    print("Starting Translation Evaluation Test")
    print("=" * 50)
    
    all_results = {}
    
    for language_pair, data in test_cases.items():
        print(f"\nTesting language pair: {language_pair}")
        print("-" * 30)
        
        try:
            # Create translation task
            task = TranslationTask(
                tokenizer=tokenizer,
                mapping_type="translation", 
                mapping_name=language_pair,
                allow_prefix=False
            )
            
            # Run comprehensive evaluation
            results = task.comprehensive_evaluate(
                sources=data["sources"],
                predictions=data["predictions"],
                references=data["references"]
            )
            
            # Print detailed results
            task.print_evaluation_details(
                sources=data["sources"],
                predictions=data["predictions"], 
                references=data["references"],
                results=results,
                num_examples=3
            )
            
            # Store results
            all_results[language_pair] = results
            
        except Exception as e:
            print(f"Error testing {language_pair}: {str(e)}")
            continue
    
    # Print summary of all results
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL LANGUAGE PAIRS")
    print("=" * 80)
    
    for lang_pair, results in all_results.items():
        print(f"{lang_pair:10} | BLEU: {results['bleu']:6.2f} | COMET: {results['comet']:6.4f}")
    
    return all_results


if __name__ == "__main__":
    results = test_translation_evaluation()
