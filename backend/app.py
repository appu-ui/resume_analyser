from flask import Flask, request, jsonify
from flask_cors import CORS
from ai_analyser import extract_text_from_pdf, analyze_resume
import sqlite3
import json

def init_db():
    conn = sqlite3.connect('resumes.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            score INTEGER NOT NULL,
            suggestions TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database table when the script loads
init_db()

app = Flask(__name__)
# Enable CORS for all routes so the frontend can make requests to the backend
CORS(app)  

@app.route('/api/analyze', methods=['POST'])
def analyze():
    # Check if a file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = file.filename.lower()
        text = ""
        
        if filename.endswith('.pdf'):
            # Extract text directly from the uploaded file stream
            text = extract_text_from_pdf(file.stream)
        elif filename.endswith('.txt'):
            # Extract text from a plain text file
            text = file.read().decode('utf-8', errors='ignore')
        else:
            return jsonify({'error': 'Unsupported file type. Please upload a PDF or TXT file.'}), 400
            
        # Call the analysis function on the extracted text
        result = analyze_resume(text)
        
        # Save the result to the SQLite database
        try:
            conn = sqlite3.connect('resumes.db')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO resumes (filename, score, suggestions) VALUES (?, ?, ?)",
                (file.filename, result.get('score', 0), json.dumps(result.get('suggestions', [])))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")

        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)