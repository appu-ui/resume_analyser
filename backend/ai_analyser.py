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

def analyze_resume(text, job_role="", experience_level=""):
    """
    Detailed AI resume analysis using Groq and Llama 3.
    """
    if not text.strip():
        return {
            "score": 0,
            "suggestions": ["The document is empty or could not be read."]
        }
        
    role_context_parts = []
    if job_role:
        role_context_parts.append(f"the role of '{job_role}'")
    if experience_level:
        role_context_parts.append(f"an experience level of '{experience_level}'")
        
    role_context = ""
    if role_context_parts:
        role_context = f" The user is applying for {' and '.join(role_context_parts)}."
        
    prompt = f"""
    You are a strict ATS resume evaluator used by recruiters.{role_context}

    You must evaluate the resume realistically. Do NOT inflate scores.

    Evaluate the resume using the following categories:

    1. Content Quality (0–25)
    - Relevant experience
    - meaningful descriptions
    - clarity of responsibilities

    2. Achievements & Impact (0–25)
    - quantified achievements
    - measurable impact
    - strong action verbs

    3. Structure & Formatting (0–20)
    - readable layout
    - logical section order
    - bullet point clarity

    4. Skills & Relevance (0–15)
    - alignment with job role
    - relevant technical skills

    5. Grammar & Clarity (0–15)
    - grammar
    - spelling
    - readability

    Total score must equal the sum of these categories (0–100).

    Important rules:
    - Be strict and realistic.
    - Most resumes should score between 55 and 75.
    - Only exceptional resumes deserve 85+.
    - If achievements are missing, reduce the score significantly.

    Return ONLY a JSON object with:

{{
"score": final_score,
"suggestions": [list of detailed improvement suggestions]
}}
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