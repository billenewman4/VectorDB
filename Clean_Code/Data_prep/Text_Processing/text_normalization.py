"""
Text normalization utilities for data preparation.

This module provides functions for standardizing and normalizing text data
for various purposes including basic cleaning and preparing text for clustering.
"""

import re
import sys
import os
from typing import Optional

# Add parent directories to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import abbreviation expansion function if available
try:
    from .abbreviation_handler import expand_abbreviations
except ImportError:
    # Raise error if original version not found
    raise ImportError("Unable to import abbreviation expansion function")


def clean_text(text: Optional[str]) -> str:
    """
    Basic text cleaning: lowercase, strip whitespace.
    
    Args:
        text: Input text to clean. Can be None or a string.
        
    Returns:
        Cleaned text string. If input is None or not a string, returns empty string.
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove excessive whitespace inside the string
    text = re.sub(r'\s+', ' ', text)
    
    return text


def preprocess_text_for_clustering(text: str, expand_abbreviations_flag: bool = True) -> str:
    """
    Enhanced preprocessing function optimized for product clustering.
    Normalizes text by expanding abbreviations, removing special characters, 
    and standardizing format.
    
    Args:
        text: Input text to preprocess
        expand_abbreviations_flag: Whether to expand abbreviations in the text
        
    Returns:
        Preprocessed text optimized for clustering
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Expand abbreviations using existing function (if enabled)
    if expand_abbreviations_flag:
        text = expand_abbreviations(text)
    
    # Standardize white space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "  HELLO  WORLD  ",
        "1# PKG ORG BNLS CHK",
        None,
        123  # Non-string input
    ]
    
    print("Testing basic text cleaning:")
    for test in test_cases:
        try:
            result = clean_text(test)
            print(f"Input: {test!r}\nOutput: {result!r}\n")
        except Exception as e:
            print(f"Error with input {test!r}: {e}")
    
    print("\nTesting preprocessing for clustering:")
    valid_tests = [t for t in test_cases if isinstance(t, str) and t is not None]
    for test in valid_tests:
        try:
            # Test with abbreviation expansion
            expanded = preprocess_text_for_clustering(test, expand_abbreviations_flag=True)
            print(f"Original: {test!r}")
            print(f"Preprocessed: {expanded!r}")
            
            # Test without abbreviation expansion
            not_expanded = preprocess_text_for_clustering(test, expand_abbreviations_flag=False)
            print(f"Preprocessed (no expansion): {not_expanded!r}\n")
        except Exception as e:
            print(f"Error processing {test!r}: {e}")
