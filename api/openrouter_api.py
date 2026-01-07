import requests
import json
import threading
from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, SYSTEM_PROMPT, GEMINI_AVAILABLE

class OpenRouterAPI:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.api_url = OPENROUTER_API_URL
        self.system_prompt = SYSTEM_PROMPT
        self.available = GEMINI_AVAILABLE
    
    def is_available(self):
        """Check if API is available."""
        return self.available and self.api_key and self.api_key != "your-api-key-here"
    
    def chat_completion(self, messages, temperature=0.7, max_tokens=1000):
        """Send request to OpenRouter API."""
        try:
            if not self.is_available():
                return None
            
            if not any(msg.get("role") == "system" for msg in messages):
                messages = [{"role": "system", "content": self.system_prompt}] + messages
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            data = {
                "model": "arcee-ai/trinity-mini:free",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            response = requests.post(
                url=self.api_url,
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"API Exception: {e}")
            return None
    
    def analyze_retina_scan(self, analysis_data):
        disc_info = f"\n- Optic Disc Diameter: {analysis_data.get('optic_disc_diameter', 0)} pixels" if analysis_data.get('optic_disc_diameter', 0) > 0 else ""
        
        prompt = f"""Analyze this retinal scan diagnosis:

DIAGNOSIS DETAILS:
- Severity: {analysis_data.get('severity', 'Unknown')} Diabetic Retinopathy
- Confidence: {analysis_data.get('confidence', 0):.1%}
- Lesions Detected: {analysis_data.get('lesion_count', 0)}
- Vessel Density: {analysis_data.get('vessel_density', 0):.2f}%
- Vessel Method: {analysis_data.get('vessel_method', 'Unknown')}{disc_info}

Please provide a brief clinical assessment (2-3 paragraphs) covering:
1. What this severity level means for the patient
2. Key clinical concerns based on the findings
3. Recommended next steps and follow-up timeline
4. Comments on vessel density and what it might indicate

Keep your response professional, concise, and clinically accurate."""
        
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages)
    
    def answer_question(self, question, context_data):
        lesion_info = ""
        if context_data.get('lesion_types'):
            lesion_info = ", ".join([f"{count} {name}" for name, count in context_data['lesion_types'].items()])
        
        disc_info = f"\n- Optic Disc Diameter: {context_data.get('optic_disc_diameter', 0)} pixels" if context_data.get('optic_disc_diameter', 0) > 0 else ""
        
        context = f"""Answer the following question about this retinal scan:

CURRENT SCAN ANALYSIS:
- Severity: {context_data.get('severity', 'Unknown')} Diabetic Retinopathy
- Confidence: {context_data.get('confidence', 0):.1%}
- Total Lesions: {context_data.get('lesion_count', 0)}
- Lesion Types: {lesion_info if lesion_info else 'None detected'}
- Vessel Density: {context_data.get('vessel_density', 0):.2f}%
- Vessel Method: {context_data.get('vessel_method', 'Unknown')}{disc_info}

USER QUESTION: {question}

Please provide a clear, professional, and clinically accurate response."""
        
        messages = [{"role": "user", "content": context}]
        return self.chat_completion(messages, temperature=0.7, max_tokens=1500)
    
    def process_in_thread(self, callback, *args, **kwargs):
        if not self.is_available():
            callback("AI Not Available", "OpenRouter API is not configured or failed to initialize.")
            return
        
        def run():
            result = None
            if len(args) > 0 and args[0] == "analyze":
                result = self.analyze_retina_scan(kwargs.get('analysis_data', {}))
            elif len(args) > 0 and args[0] == "question":
                result = self.answer_question(kwargs.get('question', ''), kwargs.get('context_data', {}))
            
            if result:
                callback(None, result)
            else:
                callback("API Error", "Failed to get response from AI service.")
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()