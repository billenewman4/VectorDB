"""
Meat Cut Abbreviation Translator

This module provides functionality to translate common meat cut abbreviations
into their full descriptions to improve product description clarity.
"""

import re


def get_abbreviation_map():
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


def expand_abbreviations(text):
    """
    Expands common food-related abbreviations in the given text to their full descriptions.
    
    Args:
        text (str): The text containing potential abbreviations.
        
    Returns:
        str: The text with abbreviations expanded to their full descriptions.
    """
    if not text or not isinstance(text, str):
        return text
        
    abbrev_map = get_abbreviation_map()
    
    # Sort abbreviations by length (longest first) to prevent partial matches
    # For example, "Bone in" should be processed before "Bone"
    sorted_abbrevs = sorted(abbrev_map.keys(), key=len, reverse=True)
    
    result = text
    
    # First pass: Handle measurement abbreviations that often appear within terms (no word boundaries)
    measurement_abbrevs = ['oz', '#', 'lb', 'lbs', 'gal', 'qt', 'pt', 'ea', 'ct', 'cs', 'dz', 'pcs', 'pc']
    for abbrev in sorted_abbrevs:
        if abbrev in measurement_abbrevs:
            # For measurements, also match when they're attached to numbers (e.g., "10oz")
            # Use lookahead/lookbehind to ensure we don't replace within other words
            pattern = r'(?i)(\d+)' + re.escape(abbrev) + r'(?![a-zA-Z])'
            replacement = r'\1 ' + abbrev_map[abbrev]
            result = re.sub(pattern, replacement, result)
    
    # Second pass: Handle all other abbreviations using word boundaries
    for abbrev in sorted_abbrevs:
        if abbrev not in measurement_abbrevs:  # Skip those already processed
            # Use word boundaries to ensure we're replacing whole words/phrases
            pattern = r'(?i)\b' + re.escape(abbrev) + r'\b'
            result = re.sub(pattern, abbrev_map[abbrev], result)
    
    return result


def expand_abbreviations_in_dataframe(df, text_columns):
    """
    Expands meat cut abbreviations in specified text columns of a DataFrame.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing product descriptions.
        text_columns (list): List of column names that contain text to process.
        
    Returns:
        pandas.DataFrame: The DataFrame with abbreviations expanded in the specified columns.
    """
    result_df = df.copy()
    
    for col in text_columns:
        if col in result_df.columns:
            result_df[col] = result_df[col].apply(expand_abbreviations)
    
    return result_df
