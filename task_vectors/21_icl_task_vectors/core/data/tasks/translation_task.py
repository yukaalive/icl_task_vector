import nltk
from transformers import PreTrainedTokenizer
from typing import Any, List, Dict, Optional
import warnings

from core import config
from core.data.tasks.mapping_task import MappingTask
from comet import download_model, load_from_checkpoint
# import sacrebleu

# Suppress warnings from COMET
warnings.filterwarnings("ignore", category=UserWarning)

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import wordnet as wn


class TranslationTask(MappingTask):
    
    def __init__(self, tokenizer: PreTrainedTokenizer, mapping_type: str, mapping_name: str, allow_prefix: bool = False):
        super().__init__(tokenizer, mapping_type, mapping_name, allow_prefix)
        
        # Initialize COMET model
        self.comet_model = None
        self._load_comet_model()
        
        # Language mapping for supported languages
        self.supported_languages = {
            "en": "eng", 
            "fr": "fra", 
            "it": "ita", 
            "es": "spa",
            "ja": "jpn"  
        }
    
    def _load_comet_model(self):
        """Load COMET model for evaluation"""
        model_path = download_model("Unbabel/wmt22-comet-da")
        self.comet_model = load_from_checkpoint(model_path)
    
    @staticmethod
    def _get_synonyms(word: str, lang_to: str):
        lang = {
            "en": "eng",
            "fr": "fra",
            "it": "ita",
            "es": "spa",
        }[lang_to]
        synonyms = [word]
        for syn in wn.synsets(word, lang=lang):
            for lemma in syn.lemmas(lang=lang):
                synonyms.append(lemma.name())
        return synonyms

    def compare_outputs(self, output1: Any, output2: Any) -> bool:
        output1, output2 = output1.strip(), output2.strip()
        output_lang = self.mapping_name.split("_")[1]
        if output_lang == "ja":
           return output1 == output2
        synonyms1 = self._get_synonyms(output1, output_lang)
        synonyms2 = self._get_synonyms(output2, output_lang)
        return len(set(synonyms1) & set(synonyms2)) > 0
    
    # def evaluate_with_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
    #     """
    #     Evaluate translations using BLEU score
    #     
    #     Args:
    #         predictions: List of predicted translations
    #         references: List of reference translations
    #         
    #     Returns:
    #         Dictionary containing BLEU scores
    #     """
    #     # Calculate BLEU score
    #     bleu_score = sacrebleu.corpus_bleu(predictions, [references])
    #     
    #     return {
    #         "bleu": bleu_score.score,
    #         "bleu_1": bleu_score.precisions[0],
    #         "bleu_2": bleu_score.precisions[1] if len(bleu_score.precisions) > 1 else 0.0,
    #         "bleu_3": bleu_score.precisions[2] if len(bleu_score.precisions) > 2 else 0.0,
    #         "bleu_4": bleu_score.precisions[3] if len(bleu_score.precisions) > 3 else 0.0,
    #     }
    
    def evaluate_with_comet(self, sources: List[str], predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Evaluate translations using COMET score
        
        Args:
            sources: List of source sentences
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            Dictionary containing COMET scores
        """
        # Prepare data for COMET
        comet_input = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]
        
        # Calculate COMET score
        comet_scores = self.comet_model.predict(comet_input, batch_size=8, gpus=0)
        print("翻訳前",sources)
        print("予測",predictions)
        print("正解",references)
        print("スコア",comet_scores)
        return {
            "comet": sum(comet_scores.scores) / len(comet_scores.scores),
            "comet_scores": comet_scores.scores
        }
    
    def comprehensive_evaluate(self, sources: List[str], predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """
        Comprehensive evaluation using both BLEU and COMET
        
        Args:
            sources: List of source sentences
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Get language information
        source_lang, target_lang = self.mapping_name.split("_")
        
        print(f"Evaluating {source_lang} -> {target_lang} translation")
        # BLEU evaluation
        # bleu_results = self.evaluate_with_bleu(predictions, references)
        
        # COMET evaluation
        comet_results = self.evaluate_with_comet(sources, predictions, references)
        
        # Combine results
        results = {
            "language_pair": f"{source_lang}-{target_lang}",
            "num_examples": len(predictions),
            # **bleu_results,
            **comet_results
        }
        
        # print(f"BLEU Score: {bleu_results['bleu']:.4f}")
        print(f"COMET Score: {comet_results['comet']:.4f}")
        
        return results
    
    def print_evaluation_details(self, sources: List[str], predictions: List[str], references: List[str], 
                               results: Dict[str, Any], num_examples: int = 10):
        """
        Print detailed evaluation results with examples
        """
        print("\n" + "="*80)
        print(f"TRANSLATION EVALUATION: {results['language_pair']}")
        print("="*80)
        
        print(f"Overall Results:")
        # print(f"  BLEU Score: {results['bleu']:.4f}")
        print(f"  COMET Score: {results['comet']:.4f}")
        print(f"  Total Examples: {results['num_examples']}")
        
        # print(f"\nDetailed BLEU Scores:")
        # print(f"  BLEU-1: {results['bleu_1']:.4f}")
        # print(f"  BLEU-2: {results['bleu_2']:.4f}")
        # print(f"  BLEU-3: {results['bleu_3']:.4f}")
        # print(f"  BLEU-4: {results['bleu_4']:.4f}")
        
        print("-" * 80)
        for i in range(min(num_examples, len(predictions))):
            comet_score = results['comet_scores'][i] if 'comet_scores' in results else 0
            print(f"Example {i+1}:")
            print(f"  Source:     {sources[i]}")
            print(f"  Reference:  {references[i]}")
            print(f"  Prediction: {predictions[i]}")
            print(f"  COMET:      {comet_score:.4f}")
            print()
