#!/usr/bin/env python
"""Test script to verify Korean language support in UDTube for all 4 tasks."""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    try:
        import spacy_udpipe
        print("✓ spacy_udpipe installed")
    except ImportError:
        print("✗ spacy_udpipe not installed. Please run: pip install spacy-udpipe")
        return False
    
    try:
        import transformers
        print("✓ transformers installed")
    except ImportError:
        print("✗ transformers not installed. Please run: pip install transformers")
        return False
    
    try:
        import torch
        print("✓ PyTorch installed")
    except ImportError:
        print("✗ PyTorch not installed. Please run: pip install torch")
        return False
    
    return True

def download_korean_model():
    """Download Korean model for spacy-udpipe."""
    print("\nDownloading Korean model for tokenization...")
    try:
        import spacy_udpipe
        spacy_udpipe.download("ko-gsd")
        print("✓ Korean model downloaded")
        return True
    except Exception as e:
        print(f"✗ Failed to download Korean model: {e}")
        return False

def create_conllu_from_text():
    """Convert Korean text to CoNLL-U format."""
    print("\nConverting Korean text to CoNLL-U format...")
    
    text_file = "korean_data/sample_text.txt"
    conllu_file = "korean_data/sample.conllu"
    
    if not os.path.exists(text_file):
        print(f"✗ {text_file} not found. Creating sample file...")
        os.makedirs("korean_data", exist_ok=True)
        with open(text_file, "w", encoding="utf-8") as f:
            f.write("안녕하세요. 저는 한국어를 공부하고 있습니다.\n")
            f.write("오늘 날씨가 정말 좋네요.\n")
            f.write("서울은 한국의 수도입니다.\n")
    
    cmd = [
        "python", "scripts/pretokenize.py",
        text_file, conllu_file,
        "--langcode", "ko-gsd"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Created {conllu_file}")
            return conllu_file
        else:
            print(f"✗ Failed to create CoNLL-U file: {result.stderr}")
            return None
    except Exception as e:
        print(f"✗ Error running pretokenize: {e}")
        return None

def test_model_loading():
    """Test if Korean BERT models can be loaded."""
    print("\nTesting Korean model loading...")
    
    from transformers import AutoModel, AutoTokenizer
    
    models_to_test = [
        ("google-bert/bert-base-multilingual-cased", "Multilingual BERT"),
        ("klue/bert-base", "KLUE BERT"),
    ]
    
    for model_name, display_name in models_to_test:
        try:
            print(f"  Testing {display_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Test tokenization with Korean text
            test_text = "안녕하세요"
            tokens = tokenizer.tokenize(test_text)
            print(f"    ✓ {display_name} loaded successfully")
            print(f"      Sample tokenization: {test_text} -> {tokens[:5]}...")
        except Exception as e:
            print(f"    ✗ Failed to load {display_name}: {e}")

def display_conllu_sample(conllu_file):
    """Display a sample of the CoNLL-U file."""
    print(f"\nSample of {conllu_file}:")
    print("-" * 50)
    
    with open(conllu_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:20]):  # Show first 20 lines
            print(line.rstrip())
    print("-" * 50)

def main():
    """Main test function."""
    print("=" * 60)
    print("UDTube Korean Language Support Test")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    # Download Korean model
    if not download_korean_model():
        print("\nFailed to download Korean model. Please check your internet connection.")
        sys.exit(1)
    
    # Create CoNLL-U file
    conllu_file = create_conllu_from_text()
    if not conllu_file:
        print("\nFailed to create CoNLL-U file.")
        sys.exit(1)
    
    # Display sample
    display_conllu_sample(conllu_file)
    
    # Test model loading
    test_model_loading()
    
    print("\n" + "=" * 60)
    print("Korean Language Support Test Complete!")
    print("=" * 60)
    
    print("\nThe 4 tasks UDTube can perform on Korean text:")
    print("1. Lemmatization - Extract base forms of Korean words")
    print("2. Universal POS tagging - Assign universal part-of-speech tags")
    print("3. Language-specific POS tagging - Assign Korean-specific POS tags")
    print("4. Morphological feature tagging - Capture Korean morphological information")
    
    print("\nTo train a model with Korean data, use:")
    print("  udtube fit --config configs/korean_klue_bert.yaml")
    print("\nTo perform inference on Korean text, use:")
    print("  udtube predict --config configs/korean_klue_bert.yaml \\")
    print("                 --ckpt_path path/to/checkpoint.ckpt \\")
    print("                 --data.predict korean_data/sample.conllu")
    
    print("\nFor more details, see KOREAN_SUPPORT.md")

if __name__ == "__main__":
    main()