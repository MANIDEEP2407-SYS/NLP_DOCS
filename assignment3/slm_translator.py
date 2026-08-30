#!/usr/bin/env python3
"""
SLM-Based English to Tamil Glossary Translator
Using Ollama with Mistral 7B (Small Language Model)
Author: Student
Date: 2026
"""

import json
import requests
from typing import List, Dict
import sys

class SLMTranslator:
    """
    A Small Language Model-based translator using Ollama.
    Mistral 7B is a compact, efficient SLM perfect for this task.
    """
    
    def __init__(self, model_name: str = "mistral", ollama_url: str = "http://localhost:11434"):
        """
        Initialize the SLM Translator
        
        Args:
            model_name: Name of the SLM model (default: mistral - 7B parameters)
            ollama_url: URL where Ollama is running
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        
        print(f"🚀 Initializing SLM Translator")
        print(f"📌 Model: {model_name} (7B - Small Language Model)")
        print(f"🔗 Ollama Server: {ollama_url}\n")
    
    def translate_term(self, english_term: str, context: str = "general") -> str:
        """
        Translate a single English term to Tamil using SLM.
        
        Args:
            english_term: The English term to translate
            context: Context for better translation (default: general)
        
        Returns:
            Tamil translation of the term
        """
        
        prompt = f"""Translate the following English technical term to Tamil. 
        Provide ONLY the Tamil translation, nothing else.
        
        Context: {context}
        English Term: {english_term}
        Tamil Translation:"""
        
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3  # Lower temperature for consistent translations
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                tamil_text = result.get("response", "").strip()
                return tamil_text
            else:
                return f"[Error: {response.status_code}]"
                
        except requests.exceptions.ConnectionError:
            return "[Error: Cannot connect to Ollama. Please ensure Ollama is running]"
        except Exception as e:
            return f"[Error: {str(e)}]"
    
    def translate_glossary(self, glossary: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """
        Translate an entire glossary from English to Tamil.
        
        Args:
            glossary: Dictionary with term -> definition structure
        
        Returns:
            Dictionary with both original and translated terms
        """
        
        translated_glossary = {}
        total_terms = len(glossary)
        
        print(f"📚 Translating {total_terms} terms using {self.model_name}...\n")
        print("=" * 70)
        print(f"{'#':<4} {'English Term':<25} {'Tamil Translation':<35}")
        print("=" * 70)
        
        for idx, (english_term, definition) in enumerate(glossary.items(), 1):
            tamil_translation = self.translate_term(english_term, context="technical")
            
            translated_glossary[english_term] = {
                "definition": definition,
                "tamil_translation": tamil_translation,
                "tamil_definition": self.translate_term(definition, context="definition")
            }
            
            # Display progress
            print(f"{idx:<4} {english_term:<25} {tamil_translation:<35}")
            sys.stdout.flush()
        
        print("=" * 70)
        print(f"\n✅ Translation Complete!\n")
        
        return translated_glossary
    
    def save_results(self, results: Dict, filename: str = "translated_glossary.json"):
        """Save translation results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Results saved to: {filename}")


# Sample glossary related to AI/ML context
SAMPLE_GLOSSARY = {
    "Algorithm": "A step-by-step procedure for solving a problem or accomplishing a task",
    "Neural Network": "A computing system inspired by biological neural networks",
    "Machine Learning": "The ability of computer systems to learn and improve from experience",
    "Data Science": "An interdisciplinary field that uses scientific methods to extract knowledge",
    "Artificial Intelligence": "The simulation of human intelligence processes by computer systems",
    "Model": "A mathematical representation of patterns learned from training data",
    "Training Data": "Sample data used to train machine learning models",
    "Parameter": "A configuration variable internal to a model whose value is learned",
    "Accuracy": "The fraction of predictions made by a model that are correct",
    "Optimization": "The process of improving model performance through iterative adjustments"
}


def main():
    """Main function to demonstrate SLM-based translation."""
    
    print("\n" + "="*70)
    print("🤖 SLM (Small Language Model) - Based Glossary Translator")
    print("="*70)
    print("Powered by: Ollama + Mistral 7B")
    print("Language Pair: English ↔ Tamil")
    print("="*70 + "\n")
    
    # Initialize translator
    translator = SLMTranslator(model_name="mistral")
    
    # Translate the sample glossary
    results = translator.translate_glossary(SAMPLE_GLOSSARY)
    
    # Save results
    translator.save_results(results)
    
    # Display formatted results
    print("\n" + "="*70)
    print("📋 TRANSLATION RESULTS - English to Tamil")
    print("="*70 + "\n")
    
    for english_term, translations in results.items():
        print(f"🔤 English Term: {english_term}")
        print(f"   📝 Definition: {translations['definition']}")
        print(f"   🇮🇳 Tamil Translation: {translations['tamil_translation']}")
        print(f"   🇮🇳 Tamil Definition: {translations['tamil_definition']}")
        print("-" * 70)
    
    print("\n✨ Translation process completed successfully!")
    print("📊 Check 'translated_glossary.json' for detailed results\n")


if __name__ == "__main__":
    main()
