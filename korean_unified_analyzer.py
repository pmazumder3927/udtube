#!/usr/bin/env python
"""
Korean Unified Morphological Analyzer
Processes Korean text and outputs a unified schema for each word including all 4 morphological tasks.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class KoreanUnifiedAnalyzer:
    """Analyzer that processes Korean text through UDTube's 4 morphological tasks."""
    
    def __init__(self, model_path: Optional[str] = None, config_path: str = "configs/korean_quick_training.yaml"):
        """
        Initialize the analyzer.
        
        Args:
            model_path: Path to trained model checkpoint (optional for training mode)
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        self.temp_dir = Path("korean_temp")
        self.temp_dir.mkdir(exist_ok=True)
        
    def text_to_conllu(self, text_file: str, output_file: str) -> bool:
        """
        Convert text file to CoNLL-U format.
        
        Args:
            text_file: Path to input text file
            output_file: Path to output CoNLL-U file
            
        Returns:
            True if successful, False otherwise
        """
        cmd = [
            "python", "scripts/pretokenize.py",
            text_file, output_file,
            "--langcode", "ko-gsd"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✓ Created CoNLL-U file: {output_file}")
                return True
            else:
                logger.error(f"Failed to create CoNLL-U: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            return False
    
    def parse_conllu(self, conllu_file: str) -> List[Dict]:
        """
        Parse CoNLL-U file into structured data.
        
        Args:
            conllu_file: Path to CoNLL-U file
            
        Returns:
            List of sentences, each containing list of word dictionaries
        """
        sentences = []
        current_sentence = []
        sentence_text = ""
        
        with open(conllu_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Handle sentence text metadata
                if line.startswith('# text'):
                    sentence_text = line.split('=', 1)[1].strip() if '=' in line else ""
                
                # Empty line marks end of sentence
                elif not line:
                    if current_sentence:
                        sentences.append({
                            'text': sentence_text,
                            'words': current_sentence
                        })
                        current_sentence = []
                        sentence_text = ""
                
                # Skip comments
                elif line.startswith('#'):
                    continue
                
                # Parse word line
                else:
                    parts = line.split('\t')
                    if len(parts) >= 10:
                        word_info = {
                            'id': parts[0],
                            'form': parts[1],
                            'lemma': parts[2] if parts[2] != '_' else None,
                            'upos': parts[3] if parts[3] != '_' else None,
                            'xpos': parts[4] if parts[4] != '_' else None,
                            'feats': parts[5] if parts[5] != '_' else None,
                        }
                        current_sentence.append(word_info)
        
        # Don't forget last sentence
        if current_sentence:
            sentences.append({
                'text': sentence_text,
                'words': current_sentence
            })
        
        return sentences
    
    def create_unified_schema(self, sentences: List[Dict]) -> Dict:
        """
        Create unified schema output for all sentences.
        
        Args:
            sentences: List of parsed sentences
            
        Returns:
            Unified schema dictionary
        """
        unified = {
            'language': 'Korean',
            'analysis_tasks': {
                '1_lemmatization': 'Extract base forms of Korean words',
                '2_upos': 'Universal part-of-speech tags',
                '3_xpos': 'Korean-specific part-of-speech tags',
                '4_features': 'Morphological features'
            },
            'sentences': []
        }
        
        for sent_idx, sentence in enumerate(sentences, 1):
            sent_data = {
                'sentence_id': sent_idx,
                'text': sentence['text'],
                'word_count': len(sentence['words']),
                'words': []
            }
            
            for word in sentence['words']:
                word_data = {
                    'position': word['id'],
                    'surface_form': word['form'],
                    'morphological_analysis': {
                        'lemma': word['lemma'] or 'N/A',
                        'universal_pos': word['upos'] or 'N/A',
                        'korean_pos': word['xpos'] or 'N/A',
                        'features': word['feats'] or 'N/A'
                    }
                }
                sent_data['words'].append(word_data)
            
            unified['sentences'].append(sent_data)
        
        # Add statistics
        total_words = sum(s['word_count'] for s in unified['sentences'])
        unified['statistics'] = {
            'total_sentences': len(sentences),
            'total_words': total_words,
            'average_words_per_sentence': round(total_words / len(sentences), 2) if sentences else 0
        }
        
        return unified
    
    def analyze_text(self, text_file: str, output_format: str = 'json') -> Union[Dict, str]:
        """
        Main analysis pipeline for Korean text.
        
        Args:
            text_file: Path to input text file
            output_format: Output format ('json', 'readable', or 'markdown')
            
        Returns:
            Analysis results in specified format
        """
        start_time = time.time()
        
        # Step 1: Convert text to CoNLL-U
        tokenization_start = time.time()
        conllu_file = self.temp_dir / "temp.conllu"
        if not self.text_to_conllu(text_file, str(conllu_file)):
            return None
        tokenization_time = time.time() - tokenization_start
        
        # Step 2: If model checkpoint exists, run prediction
        inference_time = 0
        if self.model_path and os.path.exists(self.model_path):
            predicted_file = self.temp_dir / "predicted.conllu"
            cmd = [
                "udtube", "predict",
                "--config", self.config_path,
                "--ckpt_path", self.model_path,
                "--data.predict", str(conllu_file),
                "--prediction.path", str(predicted_file),
                "--trainer.enable_progress_bar", "false"
            ]
            
            try:
                logger.info("Running UDTube prediction with trained model...")
                inference_start = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True)
                inference_time = time.time() - inference_start
                if result.returncode == 0:
                    logger.info(f"✓ Prediction completed in {inference_time:.2f}s")
                    conllu_file = predicted_file
                else:
                    logger.warning(f"Prediction failed, using tokenized data only: {result.stderr}")
            except Exception as e:
                logger.warning(f"Could not run prediction: {e}")
        
        # Step 3: Parse CoNLL-U
        sentences = self.parse_conllu(str(conllu_file))
        
        # Step 4: Create unified schema
        unified = self.create_unified_schema(sentences)
        
        # Add timing information
        total_time = time.time() - start_time
        unified['timing'] = {
            'tokenization_time': round(tokenization_time, 3),
            'inference_time': round(inference_time, 3),
            'total_time': round(total_time, 3),
            'sentences_per_second': round(len(sentences) / total_time, 2) if sentences else 0,
            'words_per_second': round(unified['statistics']['total_words'] / total_time, 2) if total_time > 0 else 0
        }
        
        # Step 5: Format output
        if output_format == 'json':
            return unified
        elif output_format == 'readable':
            return self.format_readable(unified)
        elif output_format == 'markdown':
            return self.format_markdown(unified)
        else:
            return unified
    
    def format_readable(self, unified: Dict) -> str:
        """Format unified schema as human-readable text."""
        output = []
        output.append("=" * 80)
        output.append("KOREAN TEXT UNIFIED MORPHOLOGICAL ANALYSIS")
        output.append("=" * 80)
        output.append("")
        
        # Statistics
        stats = unified['statistics']
        output.append(f"Total Sentences: {stats['total_sentences']}")
        output.append(f"Total Words: {stats['total_words']}")
        output.append(f"Average Words per Sentence: {stats['average_words_per_sentence']}")
        output.append("")
        
        # Sentences
        for sent in unified['sentences']:
            output.append(f"SENTENCE {sent['sentence_id']}: {sent['text']}")
            output.append("-" * 60)
            
            for word in sent['words']:
                output.append(f"  [{word['position']}] {word['surface_form']}")
                morph = word['morphological_analysis']
                output.append(f"      Lemma: {morph['lemma']}")
                output.append(f"      Universal POS: {morph['universal_pos']}")
                output.append(f"      Korean POS: {morph['korean_pos']}")
                output.append(f"      Features: {morph['features']}")
                output.append("")
        
        return "\n".join(output)
    
    def format_markdown(self, unified: Dict) -> str:
        """Format unified schema as Markdown."""
        output = []
        output.append("# Korean Text Unified Morphological Analysis")
        output.append("")
        
        # Statistics
        stats = unified['statistics']
        output.append("## Statistics")
        output.append(f"- **Total Sentences:** {stats['total_sentences']}")
        output.append(f"- **Total Words:** {stats['total_words']}")
        output.append(f"- **Average Words per Sentence:** {stats['average_words_per_sentence']}")
        output.append("")
        
        # Analysis Tasks
        output.append("## Analysis Tasks")
        for task_id, desc in unified['analysis_tasks'].items():
            task_name = task_id.split('_', 1)[1].title()
            output.append(f"- **{task_name}:** {desc}")
        output.append("")
        
        # Sentences
        output.append("## Detailed Analysis")
        for sent in unified['sentences']:
            output.append(f"### Sentence {sent['sentence_id']}")
            output.append(f"> {sent['text']}")
            output.append("")
            
            # Create table
            output.append("| Position | Word | Lemma | Universal POS | Korean POS | Features |")
            output.append("|----------|------|-------|---------------|------------|----------|")
            
            for word in sent['words']:
                morph = word['morphological_analysis']
                output.append(f"| {word['position']} | {word['surface_form']} | "
                            f"{morph['lemma']} | {morph['universal_pos']} | "
                            f"{morph['korean_pos']} | {morph['features']} |")
            output.append("")
        
        return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Korean Unified Morphological Analyzer - Processes Korean text through UDTube's 4 tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis (tokenization only)
  python korean_unified_analyzer.py input.txt
  
  # With trained model for full morphological analysis
  python korean_unified_analyzer.py input.txt --model checkpoint.ckpt
  
  # Different output formats
  python korean_unified_analyzer.py input.txt --format readable
  python korean_unified_analyzer.py input.txt --format markdown -o analysis.md
  
  # Using different configuration
  python korean_unified_analyzer.py input.txt --config configs/korean_kobert.yaml
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input text file containing Korean text"
    )
    
    parser.add_argument(
        "-m", "--model",
        help="Path to trained model checkpoint for full analysis (optional)"
    )
    
    parser.add_argument(
        "-c", "--config",
        default="configs/korean_klue_bert.yaml",
        help="Path to configuration file (default: configs/korean_klue_bert.yaml)"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["json", "readable", "markdown"],
        default="readable",
        help="Output format (default: readable)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path (if not specified, prints to stdout)"
    )
    
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation spaces (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = KoreanUnifiedAnalyzer(
        model_path=args.model,
        config_path=args.config
    )
    
    # Perform analysis
    logger.info(f"Analyzing: {args.input_file}")
    result = analyzer.analyze_text(args.input_file, args.format)
    
    if result is None:
        logger.error("Analysis failed")
        sys.exit(1)
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            if args.format == 'json':
                json.dump(result, f, ensure_ascii=False, indent=args.indent)
            else:
                f.write(result)
        logger.info(f"✓ Results saved to: {args.output}")
    else:
        if args.format == 'json':
            print(json.dumps(result, ensure_ascii=False, indent=args.indent))
        else:
            print(result)


if __name__ == "__main__":
    main()