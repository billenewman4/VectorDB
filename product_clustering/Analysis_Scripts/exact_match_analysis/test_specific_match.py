#!/usr/bin/env python3
"""
Test script to verify matching of specific beef tenderloin items.
This tests the modified prompt with a very specific set of items.
"""

import os
import re
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define test products
test_products = [
    (0, "40011", "Beef Tenderloin PSMO 5up Prime", "Vendor1"),
    (1, "141050", "BEEF TENDERLOIN, PSMO, 5UP, PRIME", "Vendor2"),
    (2, "13000606", "Beef Tenderloin Psmo Prime", "Vendor3"),  # Another prime tenderloin
    (3, "40010", "Beef Tenderloin PSMO 5up CHOICE", "Vendor1"),  # Different grade
    (4, "13000415", "Beef Tenderloin Psmo Choice 5 Down", "Vendor3")  # Different size
]

# Create the system message with examples
system_message = """You work for a food distributor that sells many SKUs of products. You are a product matching expert with high standards. Your task is to identify which SKUs in a list are exact matches of each other based on their descriptions/names. I will give you a cluster of SKUs (with their ID # and description) that are already grouped as similar. However I want you to be even more discerning to figure out which SKU's within the cluster are exactly the same product.

IMPORTANT RULES:
1. Products with the SAME SKU NUMBER should NEVER be matched together - this is an absolute requirement
2. Products must be the same fundamental food item, not just similar or related products
3. Be appropriately discerning in your matching - be confident in your matching decisions
4. Two products are considered exact matches if they share these critical attributes:
   - Same product type and category
   - Same brand (if specified)
   - Same size/count/weight information
   - Same key product specifications/attributes
   - Same flavor, cut, or variety of the same product

5. Minor differences that do NOT prevent a match:
   - CAPITALIZATION DIFFERENCES (uppercase vs lowercase)
   - Different ordering of words or formatting
   - Abbreviations vs. spelled-out terms ("lbs" vs "pounds")
   - Punctuation differences ("5-up" vs "5 up" vs "5up")
   - Minor typographical differences
   - Different SKU numbers (must have different SKUs to be a valid match)

6. Significant differences that DO prevent a match:
   - Different product types or categories
   - Different brands
   - Different sizes/weights/counts
   - Different key specifications
   - Different flavors, cuts, or varieties of the same product

EXAMPLES OF MATCHES (these would be considered the same product):

EXAMPLE 1:
- [SKU: 40011] Beef Tenderloin PSMO 5up Prime
- [SKU: 141050] BEEF TENDERLOIN, PSMO, 5UP, PRIME
These are matches because they are the same product (Beef Tenderloin), same preparation (PSMO), same grade (Prime), and same size specification (5up), despite capitalization and punctuation differences.

EXAMPLE 2:
- [SKU: 17864025] CHK WING WHL-JUM AMICK 40#
- [SKU: 17864016] CHK WING-z WHL-JUM SAN 40#
These are matches because they are the same product (Chicken Wings), same type (Whole Jumbo), and same weight (40 pounds), despite having different brand names.

EXAMPLE 3:
- [SKU: 240094] PORK CHOP FRENCHED, 9OZ
- [SKU: 240089] PORK CHOP FRENCHED, 8OZ
These are NOT matches because they have different weights (9oz vs 8oz), even though they are the same product type.

Look carefully at the key product specifications before making your decision. IGNORE capitalization, punctuation, and word order differences.

Output your answer as a numbered list of match groups, with each group containing the indices of products that exactly match. Use the format:
Group 1: [0, 3, 5]
Group 2: [1, 7]
...

Only include products that have at least one exact match with a DIFFERENT SKU. Products without exact matches should not be included.
If there are no exact matches in the list (which is completely valid), respond with "No exact matches found."
"""

# Create the user message
user_message = "Identify exact product matches in this test cluster:\n\n"
for idx, (_, sku, desc, _) in enumerate(test_products):
    user_message += f"{idx}: [SKU: {sku}] {desc}\n"

print("SYSTEM MESSAGE:")
print("=" * 80)
print(system_message)
print("=" * 80)
print("\nUSER MESSAGE:")
print("=" * 80)
print(user_message)
print("=" * 80)

# Call the OpenAI API
try:
    print("\nCalling OpenAI API...")
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.2  # Low temperature for more consistent results
        }
    )
    
    if response.status_code != 200:
        print(f"Error calling OpenAI API: {response.text}")
        exit(1)
        
    response_json = response.json()
    llm_response = response_json["choices"][0]["message"]["content"]
    
    print("\nLLM RESPONSE:")
    print("=" * 80)
    print(llm_response)
    print("=" * 80)
    
    # Process the LLM response to extract match groups
    match_groups = []
    
    if "No exact matches found" in llm_response:
        print("\nRESULT: No exact matches found")
    else:
        # Extract match groups using regex
        print("\nEXTRACTING MATCH GROUPS:")
        for line in llm_response.split('\n'):
            match = re.search(r'Group \d+: \[(.*?)\]', line)
            if match:
                indices_str = match.group(1)
                try:
                    indices = [int(idx.strip()) for idx in indices_str.split(',')]
                    if len(indices) >= 2:  # Only consider groups with at least 2 products
                        match_groups.append(indices)
                        print(f"Found match group: {indices}")
                        # Print the actual items that matched
                        print("MATCHED ITEMS:")
                        for idx in indices:
                            print(f"  {idx}: [SKU: {test_products[idx][1]}] {test_products[idx][2]}")
                except ValueError:
                    print(f"WARNING: Could not parse indices from line: {line}")
    
    if match_groups:
        print(f"\nRESULT: Found {len(match_groups)} match groups")
    else:
        print("\nRESULT: No valid match groups extracted")
        
except Exception as e:
    print(f"Error: {str(e)}")
