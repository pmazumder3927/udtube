#!/bin/bash

# UDTube Korean Language Quick Start Script
# This script sets up and tests Korean language support in UDTube

echo "========================================"
echo "UDTube Korean Language Quick Start"
echo "========================================"

# Step 1: Install dependencies
echo ""
echo "Step 1: Installing dependencies..."
pip install spacy-udpipe

# Step 2: Create Korean data directory
echo ""
echo "Step 2: Creating Korean data directory..."
mkdir -p korean_data

# Step 3: Create sample Korean text if it doesn't exist
if [ ! -f "korean_data/sample_text.txt" ]; then
    echo ""
    echo "Step 3: Creating sample Korean text file..."
    cat > korean_data/sample_text.txt << 'EOF'
안녕하세요. 저는 한국어를 공부하고 있습니다.
오늘 날씨가 정말 좋네요.
서울은 한국의 수도입니다.
한글은 세종대왕이 만들었습니다.
김치는 한국의 전통 음식입니다.
EOF
    echo "Created korean_data/sample_text.txt"
else
    echo ""
    echo "Step 3: Sample text file already exists"
fi

# Step 4: Convert to CoNLL-U format
echo ""
echo "Step 4: Converting Korean text to CoNLL-U format..."
python scripts/pretokenize.py korean_data/sample_text.txt korean_data/sample.conllu --langcode ko-gsd

# Step 5: Display sample output
echo ""
echo "Step 5: Sample CoNLL-U output:"
echo "----------------------------------------"
head -n 20 korean_data/sample.conllu
echo "----------------------------------------"

# Step 6: Show available configurations
echo ""
echo "Step 6: Available Korean configurations:"
echo "  - configs/korean_klue_bert.yaml (KLUE BERT - Recommended)"
echo "  - configs/korean_kobert.yaml (KoBERT)"
echo "  - configs/korean_mbert.yaml (Multilingual BERT)"

# Step 7: Instructions
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "The 4 morphological tasks UDTube performs:"
echo "1. Lemmatization - Extract base forms"
echo "2. Universal POS tagging (UPOS)"
echo "3. Language-specific POS tagging (XPOS)"
echo "4. Morphological features (FEATS)"
echo ""
echo "Next steps:"
echo ""
echo "1. To train a model on Korean data:"
echo "   udtube fit --config configs/korean_klue_bert.yaml"
echo ""
echo "2. To run inference on Korean text:"
echo "   udtube predict --config configs/korean_klue_bert.yaml \\"
echo "                  --ckpt_path path/to/checkpoint.ckpt \\"
echo "                  --data.predict korean_data/sample.conllu"
echo ""
echo "3. For detailed testing, run:"
echo "   python test_korean.py"
echo ""
echo "For more information, see KOREAN_SUPPORT.md"