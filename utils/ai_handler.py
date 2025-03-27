# utils/ai_handler.py
from google.generativeai import GenerativeModel, configure
import logging
from typing import Optional, Tuple
import os

class AIFeedbackGenerator:
    def __init__(self):
        self._configure_gemini()
        self.model = GenerativeModel('gemini-1.5-pro-latest')  # Updated model
        
    def _configure_gemini(self):
        """Configure Gemini API with enhanced safety settings and performance"""
        configure(api_key="Your Gemini key here")  # Use environment variables in production
        
        # Enhanced safety configuration for educational context
        self.safety_settings = {
            "HARASSMENT": "BLOCK_ONLY_HIGH",
            "HATE": "BLOCK_ONLY_HIGH",
            "SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
            "DANGEROUS": "BLOCK_MEDIUM_AND_ABOVE"
        }
        
        # Optimized generation config for academic feedback
        self.generation_config = {
            "temperature": 0.4,  # More focused responses
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
            "response_mime_type": "text/markdown"  # Structured output
        }

    def generate_feedback(self, text: str) -> Tuple[str, bool]:
        """Generate AI-powered academic feedback with quality flag"""
        try:
            if len(text) < 100:
                return "Submission too short for meaningful analysis", False

            prompt = f"""**Academic Feedback Request**
            Analyze this submission and provide:
            1. Content Accuracy (1-5 scale)
            2. Critical Thinking Evaluation
            3. Structural Coherence
            4. Improvement Recommendations
            5. Personalized Learning Resources
            
            **Submission Content**
            {text[:10000]}  # Increased context window
            
            Format using markdown with h2 headers and tables for ratings"""
            
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
                generation_config=self.generation_config
            )
            return response.text, True
        except Exception as e:
            logging.error(f"Gemini 1.5 Error: {str(e)}", exc_info=True)
            return "🔍 Analysis unavailable - Our AI service is temporarily offline", False

    def generate_summary_report(self, content: str) -> Optional[Tuple[str, dict]]:
        """Generate comprehensive academic report with metadata"""
        try:
            if not content.strip():
                return None

            prompt = f"""**Academic Report Generation**
            Create a detailed report from this content:
            1. Key Conceptual Understanding
            2. Knowledge Gaps Analysis
            3. Cross-Disciplinary Connections
            4. Recommended Learning Path
            
            **Content Source**
            {content[:30000]}  # Larger context handling
            
            Include metadata section with:
            - Complexity Score
            - Readability Level
            - Core Keywords"""
            
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
                generation_config=self.generation_config
            )
            
            # Parse metadata from response
            metadata = self._extract_metadata(response.text)
            return response.text, metadata
        except Exception as e:
            logging.error(f"Report Generation Failed: {str(e)}")
            return None

    def _extract_metadata(self, report: str) -> dict:
        """Parse metadata from generated reports"""
        # Implement custom parsing logic based on your format
        return {
            "complexity": 0,
            "readability": "",
            "keywords": []
        }

    def validate_content(self, text: str) -> dict:
        """Content validation using AI"""
        try:
            response = self.model.generate_content(
                f"Analyze this text for academic integrity and quality:\n{text[:5000]}"
                "Return JSON with: {valid: bool, issues: [], similarity_score: float}"
            )
            return self._parse_validation(response.text)
        except:
            return {"valid": False, "error": "Validation service unavailable"}

    def _parse_validation(self, response: str) -> dict:
        """Parse validation response"""
        # Add custom parsing implementation
        return {"valid": True, "issues": []}

def generate_secret_key() -> str:
    """Generate Flask-compatible secret key with enhanced entropy"""
    return os.urandom(32).hex()