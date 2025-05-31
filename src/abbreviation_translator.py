"""
Meat Cut Abbreviation Translator

This module provides functionality to translate common meat cut abbreviations
into their full descriptions to improve product description clarity.
"""

import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        
        # ===== MEAT INDUSTRY SPECIFIC ABBREVIATIONS =====
        
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
        
        # Regional/Source Indicators - Standardized
        'creekstone': 'creekstone',
        'angus': 'angus',
        'oma': 'omaha',
        'flat/nose off': 'flat nose off',
        'flat/nose-off': 'flat nose off',
        'nebraska': 'nebraska',
        'neb': 'nebraska',
        
        # Processing Codes - normalize to standardize
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


def expand_abbreviations(text):
    """
    Expands common food-related abbreviations in the given text to their full descriptions,
    with special handling for meat industry terminology.
    
    Args:
        text (str): The text containing potential abbreviations.
        
    Returns:
        str: The text with abbreviations expanded to their full descriptions.
    """
    if not text or not isinstance(text, str):
        return text
    
    # Lowercase for better matching    
    result = text.lower()
    
    # Log original text for debugging
    logging.debug(f"Expanding abbreviations in: {text}")
    
    # Get abbreviation mapping
    abbrev_map = get_abbreviation_map()
    
    # Sort abbreviations by length (longest first) to prevent partial matches
    # For example, "Bone in" should be processed before "Bone"
    sorted_abbrevs = sorted(abbrev_map.keys(), key=len, reverse=True)
    
    # Special handling for multi-part meat industry terms that may be separated by punctuation
    # Example: "tip on" vs "tip-on" vs "tipon"
    meat_terms = [
        (r'tip[\-\s]*on', 'lip-on'),
        (r'lip[\-\s]*on', 'lip-on'),
        (r'bone[\-\s]*in', 'bone-in'),
        (r'rib[\-\s]*eye', 'ribeye'),
        (r'roll[\-\s]*off', 'roll off'),
        (r'neck[\-\s]*off', 'neck off'),
        (r'deckle[\-\s]*off', 'deckle off'),
        (r'flat[\-\s/]*nose[\-\s]*off', 'flat nose off'),
        (r'heavy[\-\s]*weight', 'heavy weight'),
        (r'(outside|outer)[\-\s]*skirt', 'outside skirt'),
        (r'chuck[\-\s]*roll', 'chuck roll'),
        (r'chuck[\-\s]*clod', 'shoulder clod'),
        (r'shoulder[\-\s]*clod', 'shoulder clod'),
        (r'clod[\-\s]*xt', 'shoulder clod'),
        (r'chuck[\-\s]*flat', 'chuck flat'),
        (r'brisket[\-\s]*flat', 'brisket flat'),
        (r'brisket[\-\s]*at[\-\s]*code', 'brisket'),
        (r'brisket[\-\s]*deckle[\-\s]*off', 'brisket deckle off'),
        (r'teres[\-\s]*major', 'teres major'),
        (r'usda[\-\s]*(\d+[a-z]*)', r'usda \1')
    ]
    
    # Apply meat-specific patterns first
    for pattern, replacement in meat_terms:
        result = re.sub(r'\b' + pattern + r'\b', replacement, result, flags=re.IGNORECASE)
    
    # First pass: Handle measurement abbreviations that often appear within terms (no word boundaries)
    measurement_abbrevs = ['oz', '#', 'lb', 'lbs', 'gal', 'qt', 'pt', 'ea', 'ct', 'cs', 'dz', 'pcs', 'pc']
    for abbrev in sorted_abbrevs:
        if abbrev in measurement_abbrevs:
            # For measurements, also match when they're attached to numbers (e.g., "10oz")
            # Use lookahead to ensure we don't replace within other words
            pattern = r'(?i)(\d+)' + re.escape(abbrev) + r'(?![a-zA-Z])'
            replacement = r'\1 ' + abbrev_map[abbrev]
            result = re.sub(pattern, replacement, result)
    
    # Second pass: Handle all other abbreviations using word boundaries
    for abbrev in sorted_abbrevs:
        if abbrev not in measurement_abbrevs:  # Skip those already processed
            # Use word boundaries to ensure we're replacing whole words/phrases
            pattern = r'(?i)\b' + re.escape(abbrev) + r'\b'
            result = re.sub(pattern, abbrev_map[abbrev], result)
    
    # Final pass: Special handling for codes in parentheses that often indicate grades
    # For example: (ch), (ui), (uj)
    result = re.sub(r'\(ch\)', '(choice)', result, flags=re.IGNORECASE)
    result = re.sub(r'\(ui\)', '(usda inspection)', result, flags=re.IGNORECASE)
    result = re.sub(r'\(uj\)', '(usda inspection)', result, flags=re.IGNORECASE)
    
    # Clean up any double spaces created during replacements
    result = re.sub(r'\s+', ' ', result).strip()
    
    # Log the result for debugging
    if result != text.lower():
        logging.debug(f"Expanded to: {result}")
    
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
