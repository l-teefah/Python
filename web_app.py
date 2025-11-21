"""
Simple web interface for DAX to SQL converter
Run with: python web_app.py
"""

from flask import Flask, render_template_string, request, jsonify
from dax_to_sql_converter import DAXToSQLConverter
import os

app = Flask(__name__)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DAX to SQL Converter</title>
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
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            text-align: center;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        textarea, input {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            font-family: 'Courier New', monospace;
        }
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .result h3 {
            color: #333;
            margin-bottom: 15px;
        }
        .sql-output {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }
        .explanation {
            margin-top: 15px;
            padding: 15px;
            background: #e3f2fd;
            border-radius: 5px;
            color: #1976d2;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔀 DAX to SQL Converter</h1>
        <p class="subtitle">Convert your DAX measures to SQL queries using AI</p>
        
        <form id="converterForm">
            <div class="form-group">
                <label for="dax">DAX Measure:</label>
                <textarea id="dax" name="dax" placeholder="Enter your DAX measure here, e.g., SUM(Sales[Amount])" required></textarea>
            </div>
            
            <div class="form-group">
                <label for="table">Table Name (optional):</label>
                <input type="text" id="table" name="table" placeholder="e.g., Sales">
            </div>
            
            <button type="submit" id="convertBtn">Convert to SQL</button>
        </form>
        
        <div id="result"></div>
    </div>
    
    <script>
        document.getElementById('converterForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btn = document.getElementById('convertBtn');
            const resultDiv = document.getElementById('result');
            const dax = document.getElementById('dax').value;
            const table = document.getElementById('table').value;
            
            btn.disabled = true;
            btn.textContent = 'Converting...';
            resultDiv.innerHTML = '<div class="loading">🔄 Converting your DAX measure...</div>';
            
            try {
                const response = await fetch('/convert', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ dax, table })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    resultDiv.innerHTML = `<div class="error"><strong>Error:</strong> ${data.error}</div>`;
                } else {
                    let html = '<div class="result">';
                    html += '<h3>📊 SQL Query:</h3>';
                    html += `<div class="sql-output">${data.sql}</div>`;
                    if (data.explanation) {
                        html += `<div class="explanation"><strong>Explanation:</strong> ${data.explanation}</div>`;
                    }
                    html += '</div>';
                    resultDiv.innerHTML = html;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error"><strong>Error:</strong> ${error.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Convert to SQL';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/convert', methods=['POST'])
def convert():
    try:
        data = request.json
        dax_measure = data.get('dax', '').strip()
        table_name = data.get('table', '').strip() or None
        
        if not dax_measure:
            return jsonify({'error': 'DAX measure is required'}), 400
        
        # Initialize converter
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return jsonify({'error': 'OPENAI_API_KEY environment variable is not set'}), 500
        
        converter = DAXToSQLConverter(api_key=api_key, model='gpt-4')
        result = converter.convert(dax_measure, table_name=table_name)
        
        return jsonify({
            'sql': result.sql_query,
            'explanation': result.explanation,
            'confidence': result.confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting DAX to SQL Converter Web App...")
    print("Make sure OPENAI_API_KEY is set in your environment")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)

