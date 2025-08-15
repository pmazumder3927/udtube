#!/usr/bin/env python
"""
Korean Text Analyzer Web Interface
A simple web interface for analyzing Korean text with UDTube's unified morphological schema.
"""

import json
import os
import tempfile
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
from korean_unified_analyzer import KoreanUnifiedAnalyzer

app = Flask(__name__)

# HTML template with embedded CSS and JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Korean Text Unified Analyzer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .content {
            padding: 30px;
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        .input-section h2 {
            color: #2d3748;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        textarea {
            width: 100%;
            min-height: 150px;
            padding: 15px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 16px;
            font-family: 'Malgun Gothic', sans-serif;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        button {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button.secondary {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        }
        
        .format-selector {
            display: inline-block;
            margin-left: 10px;
        }
        
        select {
            padding: 12px 20px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: border-color 0.3s;
        }
        
        select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .results-section {
            margin-top: 30px;
            display: none;
        }
        
        .results-section h2 {
            color: #2d3748;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .stats {
            background: #f7fafc;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .timing-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        
        .timing-item {
            text-align: center;
            padding: 10px;
        }
        
        .timing-value {
            font-size: 2em;
            font-weight: bold;
            display: block;
        }
        
        .timing-label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 5px;
        }
        
        .stats h3 {
            color: #4a5568;
            margin-bottom: 10px;
        }
        
        .stat-item {
            display: inline-block;
            margin-right: 30px;
            padding: 10px 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 10px;
        }
        
        .stat-label {
            color: #718096;
            font-size: 0.9em;
        }
        
        .stat-value {
            color: #2d3748;
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .sentence {
            background: #f7fafc;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .sentence-header {
            color: #2d3748;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        
        .sentence-text {
            color: #4a5568;
            font-style: italic;
            margin-bottom: 15px;
            padding: 10px;
            background: white;
            border-left: 4px solid #667eea;
            border-radius: 5px;
        }
        
        .word-analysis {
            margin-top: 15px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        th {
            background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        tr:hover {
            background: #f7fafc;
        }
        
        tr:last-child td {
            border-bottom: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #fed7d7;
            color: #742a2a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
        }
        
        .json-output {
            background: #2d3748;
            color: #48bb78;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            overflow-x: auto;
            white-space: pre;
            display: none;
        }
        
        .example-text {
            color: #667eea;
            cursor: pointer;
            text-decoration: underline;
            margin-left: 10px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🇰🇷 Korean Text Unified Analyzer</h1>
            <p>Comprehensive morphological analysis using UDTube's 4 tasks</p>
        </div>
        
        <div class="content">
            <div class="input-section">
                <h2>Input Korean Text 
                    <span class="example-text" onclick="loadExample()">Load Example</span>
                </h2>
                <textarea id="inputText" placeholder="한국어 텍스트를 입력하세요...">안녕하세요. 저는 한국어를 공부하고 있습니다.
오늘 날씨가 정말 좋네요.
서울은 한국의 수도입니다.</textarea>
                
                <div class="button-group">
                    <button onclick="analyzeText()">Analyze Text</button>
                    <button class="secondary" onclick="clearAll()">Clear</button>
                    <div class="format-selector">
                        <label for="format">Output Format:</label>
                        <select id="format">
                            <option value="table">Table View</option>
                            <option value="json">JSON</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="loading">
                <div class="spinner"></div>
                <p>Analyzing Korean text...</p>
            </div>
            
            <div class="error" id="errorBox"></div>
            
            <div class="results-section" id="results">
                <h2>Analysis Results</h2>
                
                <div class="stats" id="statistics"></div>
                
                <div id="tableOutput"></div>
                <pre class="json-output" id="jsonOutput"></pre>
            </div>
        </div>
    </div>
    
    <script>
        function loadExample() {
            const exampleText = `안녕하세요. 저는 한국어를 공부하고 있습니다.
오늘 날씨가 정말 좋네요.
서울은 한국의 수도입니다.
한글은 세종대왕이 만들었습니다.
김치는 한국의 전통 음식입니다.`;
            document.getElementById('inputText').value = exampleText;
        }
        
        function clearAll() {
            document.getElementById('inputText').value = '';
            document.getElementById('results').style.display = 'none';
            document.getElementById('errorBox').style.display = 'none';
        }
        
        async function analyzeText() {
            const text = document.getElementById('inputText').value.trim();
            if (!text) {
                showError('Please enter Korean text to analyze');
                return;
            }
            
            const format = document.getElementById('format').value;
            
            // Show loading
            document.querySelector('.loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('errorBox').style.display = 'none';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                if (!response.ok) {
                    throw new Error('Analysis failed');
                }
                
                const data = await response.json();
                displayResults(data, format);
                
            } catch (error) {
                showError('Error: ' + error.message);
            } finally {
                document.querySelector('.loading').style.display = 'none';
            }
        }
        
        function displayResults(data, format) {
            document.getElementById('results').style.display = 'block';
            
            // Display timing information if available
            if (data.timing) {
                const timing = data.timing;
                const timingHtml = `
                    <div class="timing-info">
                        <div class="timing-item">
                            <span class="timing-value">${timing.inference_time}s</span>
                            <div class="timing-label">Inference Time</div>
                        </div>
                        <div class="timing-item">
                            <span class="timing-value">${timing.total_time}s</span>
                            <div class="timing-label">Total Time</div>
                        </div>
                        <div class="timing-item">
                            <span class="timing-value">${timing.words_per_second}</span>
                            <div class="timing-label">Words/Second</div>
                        </div>
                        <div class="timing-item">
                            <span class="timing-value">${timing.sentences_per_second}</span>
                            <div class="timing-label">Sentences/Second</div>
                        </div>
                    </div>
                `;
                document.getElementById('statistics').innerHTML = timingHtml;
            }
            
            // Display statistics
            const stats = data.statistics;
            document.getElementById('statistics').innerHTML += `
                <h3>Statistics</h3>
                <div class="stat-item">
                    <div class="stat-label">Total Sentences</div>
                    <div class="stat-value">${stats.total_sentences}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Total Words</div>
                    <div class="stat-value">${stats.total_words}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Avg Words/Sentence</div>
                    <div class="stat-value">${stats.average_words_per_sentence}</div>
                </div>
            `;
            
            if (format === 'json') {
                document.getElementById('tableOutput').style.display = 'none';
                document.getElementById('jsonOutput').style.display = 'block';
                document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
            } else {
                document.getElementById('jsonOutput').style.display = 'none';
                document.getElementById('tableOutput').style.display = 'block';
                
                let html = '';
                for (const sentence of data.sentences) {
                    html += `
                        <div class="sentence">
                            <div class="sentence-header">Sentence ${sentence.sentence_id}</div>
                            <div class="sentence-text">${sentence.text}</div>
                            <div class="word-analysis">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Position</th>
                                            <th>Word</th>
                                            <th>Lemma</th>
                                            <th>Universal POS</th>
                                            <th>Korean POS</th>
                                            <th>Features</th>
                                        </tr>
                                    </thead>
                                    <tbody>`;
                    
                    for (const word of sentence.words) {
                        const morph = word.morphological_analysis;
                        html += `
                            <tr>
                                <td>${word.position}</td>
                                <td><strong>${word.surface_form}</strong></td>
                                <td>${morph.lemma}</td>
                                <td>${morph.universal_pos}</td>
                                <td>${morph.korean_pos}</td>
                                <td>${morph.features}</td>
                            </tr>`;
                    }
                    
                    html += `
                                    </tbody>
                                </table>
                            </div>
                        </div>`;
                }
                
                document.getElementById('tableOutput').innerHTML = html;
            }
        }
        
        function showError(message) {
            const errorBox = document.getElementById('errorBox');
            errorBox.textContent = message;
            errorBox.style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(text)
            temp_file = f.name
        
        try:
            # Initialize analyzer with trained model
            analyzer = KoreanUnifiedAnalyzer(
                model_path="korean_models/lightning_logs/version_0/checkpoints/korean-quick-epoch=002-val_loss=2.0263.ckpt",
                config_path="configs/korean_quick_training.yaml"
            )
            
            # Perform analysis
            result = analyzer.analyze_text(temp_file, output_format='json')
            
            if result is None:
                return jsonify({'error': 'Analysis failed'}), 500
            
            return jsonify(result)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Korean Text Analyzer Web Interface...")
    print("Open your browser and navigate to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)