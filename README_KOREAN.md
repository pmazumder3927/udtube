# UDTube with Korean Language Support

UDTube is a neural morphological analyzer that performs 4 linguistic tasks on text. This repository has been extended with full Korean language support.

## Quick Start

### Installation
```bash
# Install UDTube and dependencies
pip install .
pip install spacy-udpipe

# Quick test
python test_korean.py
```

### Running the Web Interface
```bash
python korean_web_interface.py
# Open http://localhost:5000
```

### Command Line Usage
```bash
# Analyze Korean text
python korean_unified_analyzer.py korean_text.txt --format json

# Use trained model for full analysis
python korean_unified_analyzer.py korean_text.txt \
  --model korean_models/lightning_logs/version_0/checkpoints/korean-quick-epoch=002-val_loss=2.0263.ckpt
```

## The 4 Morphological Tasks

1. **Lemmatization** - Extracts base word forms
   - Example: 먹었다 (ate) → 먹다 (eat)

2. **Universal POS Tagging (UPOS)** - Language-independent tags
   - Example: 책 → NOUN, 읽다 → VERB

3. **Language-specific POS Tagging (XPOS)** - Korean-specific tags
   - Example: NNG (general noun), VV (verb), JKS (subject particle)

4. **Morphological Features** - Grammatical properties
   - Example: Tense=Past, Honorific=Form

## Supported Korean Models

| Model | Description | Best For |
|-------|-------------|----------|
| **KLUE BERT** | Trained on 62GB Korean text | Best Korean accuracy |
| **KoBERT** | Korean-specific BERT | Good alternative |
| **mBERT** | Multilingual (104 languages) | Mixed language needs |

## Training Your Own Model

### Quick Training (3 epochs, ~1 minute)
```bash
udtube fit --config configs/korean_quick_training.yaml
```

### Full Training (20 epochs, better accuracy)
```bash
udtube fit --config configs/korean_training.yaml
```

### Inference with Trained Model
```bash
udtube predict --config configs/korean_quick_training.yaml \
  --ckpt_path path/to/checkpoint.ckpt \
  --data.predict korean_text.conllu
```

## Configuration Files

- `configs/korean_quick_training.yaml` - Quick 3-epoch training with mBERT
- `configs/korean_training.yaml` - Full 20-epoch training  
- `configs/korean_klue_bert.yaml` - KLUE BERT configuration
- `configs/korean_kobert.yaml` - KoBERT configuration
- `configs/korean_mbert.yaml` - Multilingual BERT configuration

## Performance

With quick training (3 epochs):
- UPOS Accuracy: 92.5%
- XPOS Accuracy: 80.1%
- Lemma Accuracy: 83.7%
- Features Accuracy: 99.7%

With full training, expect 96-98% accuracy.

## File Structure

```
udtube/
├── configs/                    # Configuration files
│   ├── korean_*.yaml           # Korean-specific configs
├── korean_data/                # Sample Korean text
│   └── sample_text.txt
├── korean_models/              # Trained models (gitignored)
├── korean_unified_analyzer.py  # CLI analyzer
├── korean_web_interface.py     # Web interface
├── korean_quickstart.sh        # Setup script
└── test_korean.py              # Test script
```

## Web Interface Features

- Real-time Korean text analysis
- Inference time display
- Interactive table view
- JSON export
- Performance metrics (words/second, sentences/second)

## Converting Text to CoNLL-U

```bash
# For Korean text
python scripts/pretokenize.py input.txt output.conllu --langcode ko-gsd
```

## Troubleshooting

1. **Memory issues**: Reduce `batch_size` in config
2. **Tokenization errors**: Ensure UTF-8 encoding
3. **Model loading**: For KoBERT, may need `trust_remote_code=True`

## License

Apache 2.0 (see LICENSE.txt)

## Acknowledgments

- Original UDTube by CUNY-CL
- Korean models: KLUE, SKTBrain (KoBERT)
- Training data: Universal Dependencies Korean treebanks