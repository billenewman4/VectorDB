"""
GPT-4 based selector for choosing the best USDA code from top embedding matches.
"""

import os
import openai
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import src.config as config

class GPT4Selector:
    """Uses GPT-4 to select the best USDA code from top embedding matches."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """Initialize the GPT-4 selector with API key."""
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set it in environment or pass directly.")
            
        # Initialize OpenAI client with current API approach, without proxy settings
        try:
            # First attempt - without proxies
            self.client = openai.OpenAI(api_key=self.api_key)
        except TypeError as e:
            if 'proxies' in str(e):
                # Try again using a different approach to avoid proxy settings
                import os
                original_http_proxy = os.environ.pop('HTTP_PROXY', None)
                original_https_proxy = os.environ.pop('HTTPS_PROXY', None)
                
                try:
                    self.client = openai.OpenAI(api_key=self.api_key)
                finally:
                    # Restore environment variables
                    if original_http_proxy:
                        os.environ['HTTP_PROXY'] = original_http_proxy
                    if original_https_proxy:
                        os.environ['HTTPS_PROXY'] = original_https_proxy
            else:
                # Re-raise the error if it's not related to proxies
                raise
                
        self.model = model
        print(f"Initialized GPT-4 selector with model: {model}")
        
    def select_best_match(self, 
                         product_description: str, 
                         candidate_usda_codes: List[Tuple[str, float]]) -> Tuple[str, float, str]:
        """
        Select the best USDA code match from candidates using GPT-4.
        
        Args:
            product_description: The product description to match
            candidate_usda_codes: List of (usda_code, similarity_score) tuples
            
        Returns:
            Tuple of (selected_usda_code, confidence_score, reasoning)
        """
        print(f"Using LLM to evaluate best match for: '{product_description}'")
        print(f"Evaluating {len(candidate_usda_codes)} candidate USDA codes")
        
        if not candidate_usda_codes:
            return None, 0.0, "No candidate USDA codes provided"
        
        # Always have a fallback based on similarity
        best_match_by_similarity = max(candidate_usda_codes, key=lambda x: x[1])
        
        # Format the candidates for the prompt
        candidates_text = "\n".join([
            f"{i+1}. {code} (similarity score: {score:.4f})"
            for i, (code, score) in enumerate(candidate_usda_codes)
        ])
        
        # Construct the prompt
        prompt = f"""
You are a food product classification expert. Your task is to select the most appropriate USDA code for a given food product description.

PRODUCT DESCRIPTION: {product_description}

TOP CANDIDATE USDA CODES (with similarity scores from embedding model):
{candidates_text}

Based on your expertise in food products and USDA classification standards, which of these USDA codes is the MOST appropriate match for this product description?

Provide your answer in the following format:
SELECTED CODE: [the selected USDA code]
CONFIDENCE: [score between 0 and 1]
REASONING: [brief explanation of your selection]
"""

        # Use direct requests instead of the OpenAI client
        try:
            import requests
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a meat product classification expert."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 500
            }
            
            api_url = "https://api.openai.com/v1/chat/completions"
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            response_text = response_data["choices"][0]["message"]["content"]
            
            # Parse the response
            try:
                # Extract the selected code from the response
                selected_code_line = [line for line in response_text.split('\n') if line.startswith('SELECTED CODE:')]
                selected_code = selected_code_line[0].replace('SELECTED CODE:', '').strip() if selected_code_line else None
                
                # Extract the confidence from the response
                confidence_line = [line for line in response_text.split('\n') if line.startswith('CONFIDENCE:')]
                confidence = float(confidence_line[0].replace('CONFIDENCE:', '').strip()) if confidence_line else 0.0
                
                # Extract the reasoning
                reasoning_parts = response_text.split('REASONING:')
                reasoning = reasoning_parts[1].strip() if len(reasoning_parts) > 1 else "No reasoning provided"
                
                # Verify the selected code is one of the candidates
                candidate_codes = [code for code, _ in candidate_usda_codes]
                if selected_code not in candidate_codes:
                    print(f"LLM selected '{selected_code}' which is not in candidates. Defaulting to highest similarity.")
                    return best_match_by_similarity[0], best_match_by_similarity[1], f"Selected code not in candidates. {reasoning}"
                
                print(f"LLM selected USDA code: {selected_code} with confidence {confidence:.2f}")
                return selected_code, confidence, reasoning
                
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                return best_match_by_similarity[0], best_match_by_similarity[1], f"Error parsing response: {e}"
                
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return best_match_by_similarity[0], best_match_by_similarity[1], f"API error: {e}"
