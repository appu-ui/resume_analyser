import os
import json
from groq import Groq
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

# Initializing Groq client with the API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def extract_text_from_pdf(file_stream):
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def analyze_resume(text):
    """
    Detailed AI resume analysis using Groq and Llama 3.
    """
    if not text.strip():
        return {
            "score": 0,
            "suggestions": ["The document is empty or could not be read."]
        }
        
    prompt = f"""
    You are an expert AI resume analyzer and career coach.
    Please review the following resume text and provide a highly detailed, constructive analysis.
    Return your analysis as a JSON object with exactly two keys:
    1. "score": An integer from 0 to 100 representing the overall quality, impact, and formatting of the resume. 
    2. "suggestions": A list of strings, providing specific, detailed, and actionable suggestions to improve the resume. Include feedback on phrasing, missing sections, formatting, and impactful verbs.

    Resume Text:
    {text}
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized API that outputs only raw JSON. Do not use markdown formatting, do not wrap your response in backticks or code blocks. Output the raw JSON object containing 'score' and 'suggestions' directly."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        
        result_str = response.choices[0].message.content
        result_json = json.loads(result_str)
        
        score = result_json.get("score", 50)
        suggestions = result_json.get("suggestions", [])
        
        # In case the AI returns a string instead of a list for suggestions
        if isinstance(suggestions, str):
            suggestions = [suggestions]
            
        if not suggestions:
            suggestions = ["No specific suggestions were generated. Consider reviewing basic formatting and action verbs."]
            
        return {
            "score": score,
            "suggestions": suggestions
        }
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return {
            "score": 0,
            "suggestions": [f"Error analyzing resume with AI. Please try again later. Details: {str(e)}"]
        }