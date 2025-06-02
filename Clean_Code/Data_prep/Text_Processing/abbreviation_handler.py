"""
Abbreviation handling module for text normalization.

This module provides functionality to expand common food-related abbreviations
in text and dataframes to improve clarity and standardization of product descriptions.
"""

import re
import pandas as pd
from typing import Dict, List, Optional


def get_abbreviation_map() -> Dict[str, str]:
    """
    Returns a dictionary mapping common food-related abbreviations to their full descriptions.
    
    Returns:
        dict: A dictionary of abbreviation-to-description mappings.
    """
    return {
        # Meat cut abbreviations
        'Bn-in': 'Bone in',
        'Bnls': 'Boneless',
        'Bnl': 'Boneless',
        'Cntr Cut': 'Center Cut',
        'Cov': 'Cover',
        'Dkle': 'Deckle',
        'Dfatd': 'Defatted',
        'Dnd': 'Denuded',
        'Dia': 'Diamond',
        'Div': 'Divided',
        'Ex': 'Extra',
        'Fr': 'Fresh',
        'Frz': 'Frozen',
        'Grnd': 'Ground',
        'Inter': 'Intermediate',
        'IM': 'Individual Muscle',
        'Nk-off': 'Neck off',
        'NTE': 'Not to Exceed',
        'Oven-Prep': 'Oven-Prepared',
        'Part': 'Partially',
        'Pld': 'Peeled',
        'Prthse': 'Porterhouse',
        'Portn': 'Portion',
        'Reg': 'Regular',
        'Rst-Rdy': 'Roast-Ready',
        'Rst': 'Roast',
        'Rnd': 'Round',
        'Sh Cut': 'Short Cut',
        'Shld': 'Shoulder',
        'Sirln': 'Sirloin',
        'Sknd': 'Skinned',
        'Sp': 'Special',
        'Sq-Cut': 'Square Cut',
        'Stk': 'Steak',
        'Tender': 'Tenderloin',
        'Tri Tip': 'Triangle Tip',
        'Trmd': 'Trimmed',
        'Untrmd': 'Untrimmed',
        
        # Packaging and measurement abbreviations
        'oz': 'ounce',
        '#': 'pound',
        'lb': 'pound',
        'lbs': 'pounds',
        'gal': 'gallon',
        'qt': 'quart',
        'pt': 'pint',
        'fl oz': 'fluid ounce',
        'pkg': 'package',
        'pkgs': 'packages',
        'cnt': 'container',
        'ea': 'each',
        'pcs': 'pieces',
        'pc': 'piece',
        'ct': 'count',
        'cs': 'case',
        'dz': 'dozen',
        
        # Food preparation abbreviations
        'chk': 'chicken',
        'chx': 'chicken',
        'ckn': 'chicken',
        'ck': 'chicken',
        'tur': 'turkey',
        'bf': 'beef',
        'pk': 'pork',
        'vl': 'veal',
        'lmb': 'lamb',
        'veg': 'vegetable',
        'vegt': 'vegetable',
        'vgts': 'vegetables',
        'tom': 'tomato',
        'toms': 'tomatoes',
        'pot': 'potato',
        'pots': 'potatoes',
        'chs': 'cheese',
        'chdr': 'cheddar',
        'mozz': 'mozzarella',
        'org': 'organic',
        'nat': 'natural',
        'whl': 'whole',
        'slc': 'slice',
        
        # Meat Grade Terminology
        'ch': 'choice',
        'cho': 'choice',
        'chce': 'choice',
        'sel': 'select',
        'prm': 'prime',
        'pr': 'prime',
        
        # Bone-related Terminology
        'bny': 'bone-in',
        'bi': 'bone-in',
        'bn': 'bone',
        'bn-in': 'bone-in',
        'bnls': 'boneless',
        'bnlss': 'boneless',
        'bnl': 'boneless',
        'bonlss': 'boneless',
        
        # Cut Style Terminology
        'lip on': 'lip-on',
        'lip-on': 'lip-on',
        'tip on': 'lip-on',  # Standardizing 'tip on' to 'lip-on'
        'tipon': 'lip-on',  # Standardizing 'tipon' to 'lip-on'
        'roll-off': 'roll off',
        'roll off': 'roll off',
        'necked': 'neck off',
        'neckoff': 'neck off',
        'neck-off': 'neck off',
        'deckle off': 'deckle off',
        'dkle off': 'deckle off',
        
        # Cut Names and Variations
        'rib eye': 'ribeye',
        'rib-eye': 'ribeye',
        'rbeye': 'ribeye',
        'rbey': 'ribeye',
        'rby': 'ribeye',
        'ribeye': 'ribeye',
        'r eye': 'ribeye',
        'hvw': 'heavy weight',
        'hvw upon': 'heavy weight ribeye',
        'hvy': 'heavy weight',
        'hvy upon': 'heavy weight ribeye',
        'hw': 'heavy weight',
        
        'chuck roll': 'chuck roll',
        'chuck clod': 'shoulder clod',
        'shoulder clod': 'shoulder clod',
        'clod': 'shoulder clod',
        'clod xt': 'shoulder clod',
        'chuck flat': 'chuck flat',
        
        'brisket flat': 'brisket flat',
        'brisket at code': 'brisket',
        'brisket deckle off': 'brisket deckle off',
        
        'outside skirt': 'outside skirt',
        'outside skrt': 'outside skirt',
        'skrt': 'skirt',
        
        'teres major': 'teres major',
        
        # Regional/Source Indicators
        'creekstone': 'creekstone',
        'angus': 'angus',
        'oma': 'omaha',
        'flat/nose off': 'flat nose off',
        'flat/nose-off': 'flat nose off',
        'nebraska': 'nebraska',
        'neb': 'nebraska',
        
        # Processing Codes
        '1/4': 'quarter',
        
        # State/Form Indicators
        'frzn': 'frozen',
        'frz': 'frozen',
        'fr': 'fresh',
        'slcs': 'slices',
        'slcd': 'sliced',
        'pud': 'peeled and deveined',
        't/off': 'tail off',
        'ez': 'easy',
        'wht': 'white',
        'grn': 'green',
        'blk': 'black',
        'brn': 'brown',
        'med': 'medium',
        'lg': 'large',
        'sm': 'small',
        'xl': 'extra large',
        'xsm': 'extra small',
        'kc': 'Kansas City',
        'ny': 'New York',
    }


def expand_abbreviations(text: Optional[str]) -> str:
    """
    Expands common food-related abbreviations in the given text to their full descriptions,
    with special handling for meat industry terminology.
    
    Args:
        text: The text containing potential abbreviations. Can be None.
        
    Returns:
        str: The text with abbreviations expanded to their full descriptions.
              If input is None, returns an empty string.
    """
    if not text:
        return ""
        
    # Get abbreviation map
    abbr_map = get_abbreviation_map()
    
    # Convert to lowercase for consistent processing
    text = text.lower()
    
    # Create a function to replace abbreviations
    def replace_abbr(match):
        abbr = match.group(0).lower()
        # Return the expanded form if it exists, otherwise return the original
        return abbr_map.get(abbr, abbr)
    
    # Create a regex pattern that matches any of the abbreviations
    # Sort by length (longest first) to avoid partial matches
    abbr_keys = sorted(abbr_map.keys(), key=len, reverse=True)
    pattern = r'\b(' + '|'.join(map(re.escape, abbr_keys)) + r')\b'
    
    # Replace all abbreviations
    expanded_text = re.sub(pattern, replace_abbr, text, flags=re.IGNORECASE)
    
    return expanded_text


def expand_abbreviations_in_dataframe(df: pd.DataFrame, 
                                    text_columns: List[str]) -> pd.DataFrame:
    """
    Expands abbreviations in specified text columns of a DataFrame.
    
    Args:
        df: The DataFrame containing product descriptions.
        text_columns: List of column names that contain text to process.
        
    Returns:
        DataFrame with abbreviations expanded in the specified columns.
    """
    if df is None or df.empty:
        return df
        
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Process each specified column
    for col in text_columns:
        if col in result_df.columns:
            # Apply abbreviation expansion to the column
            result_df[col] = result_df[col].apply(expand_abbreviations)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame")
    
    return result_df


if __name__ == "__main__":
    # Create a sample DataFrame for testing
    data = {
        'product_code': ['A123', 'B456', 'C789'],
        'product_description': ['1# pkg bnls chk', '8 oz bf grnd', '12 oz frz veg']
    }
    df = pd.DataFrame(data)
    
    # Test abbreviation expansion on individual strings
    test_strings = [
        "1# pkg org bnls chk",
        "10 oz bf grnd",
        "fresh rbey stk 12 oz",
        "2 lb pkg frz veg"
    ]
    
    print("Testing abbreviation expansion on strings:")
    for test in test_strings:
        expanded = expand_abbreviations(test)
        print(f"Original: {test}")
        print(f"Expanded: {expanded}")
        print()
    
    # Test abbreviation expansion on DataFrame
    print("\nTesting abbreviation expansion on DataFrame:")
    print("Original DataFrame:")
    print(df)
    
    expanded_df = expand_abbreviations_in_dataframe(df, ['product_description'])
    print("\nExpanded DataFrame:")
    print(expanded_df)
