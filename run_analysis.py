#!/usr/bin/env python3
"""
Interactive analysis script for VectorDB product matching.
This script allows the user to select different options for analysis:
1. Embedding type (OpenAI vs other local models)
2. Data scope (limited products, only USDA coded products, or all products)
3. Optional LLM verification of matches
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.VectorDB.OpenAIEmbedder import OpenAIEmbedder
try:
    from src.VectorDB.localEmbedder import LocalEmbedder
except ImportError:
    print("Warning: LocalEmbedder import failed - local embedding may not be available")

try:
    from src.llm_selector import GPT4Selector
except ImportError:
    print("Warning: GPT4Selector import failed - LLM verification may not be available")

import src.config as config
from src.data_processing import load_transaction_data, process_transaction_data

# Helper function to get USDA code lookup map
def get_usda_lookup():
    """Build a lookup map from product codes to USDA codes."""
    import re
    # Helper function to normalize IDs from the mapping file
    def normalize_mapping_id(code):
        if isinstance(code, str):
            # Remove trailing '-<number>' and strip whitespace
            code = re.sub(r'-\d+$', '', code).strip()
            # Also remove any leading company prefix (single digit followed by digits)
            code = re.sub(r'^\d(\d+)$', r'\1', code)
        else:
            code = str(code).strip()  # Convert non-strings to string and strip
        return code

    # Load mapping file
    try:
        mapping_df = pd.read_excel(
            config.GROUND_TRUTH_FILE, 
            sheet_name=config.GROUND_TRUTH_SHEET_NAME
        )
        
        # Build lookup from each ID column to USDA code
        lookup_map = {}
        for id_col in config.GROUND_TRUTH_ID_COLS:
            if id_col in mapping_df.columns:
                # For each row, add mapping from normalized ID to USDA code
                for _, row in mapping_df.iterrows():
                    if pd.notna(row[id_col]) and pd.notna(row[config.GROUND_TRUTH_USDA_COL]):
                        norm_id = normalize_mapping_id(row[id_col])
                        usda_code = str(row[config.GROUND_TRUTH_USDA_COL]).strip()
                        lookup_map[norm_id] = usda_code
        
        print(f"Built USDA lookup map with {len(lookup_map)} entries")
        return lookup_map
    except Exception as e:
        print(f"Error building USDA lookup map: {e}")
        return {}

# Main analysis function
def run_analysis(
    embedding_type: str, 
    data_scope: str, 
    limit: Optional[int] = None,
    use_llm_verify: bool = False,
    k_samples: int = 5
):
    """
    Run vector database analysis with the specified options.
    
    Args:
        embedding_type: 'openai' or 'local'
        data_scope: 'limited', 'usda_only', or 'all'
        limit: Maximum number of products to process (for 'limited' scope)
        use_llm_verify: Whether to use LLM verification
        k_samples: Number of samples to use for LLM verification
    """
    print(f"\n{'='*80}")
    print(f"Running analysis with: Embedding={embedding_type}, Scope={data_scope}, LLM Verify={use_llm_verify}")
    
    # 1. Load and process transaction data
    print("\n1. Loading transaction data...")
    raw_data = load_transaction_data()
    if raw_data is None:
        print("Error: Failed to load transaction data")
        return
    
    unique_products_df = process_transaction_data(raw_data)
    if unique_products_df is None or len(unique_products_df) == 0:
        print("Error: No products found after processing")
        return
    
    # 2. Apply data scope filters
    print(f"\n2. Applying data scope: {data_scope}")
    
    # Get USDA mapping
    usda_lookup = get_usda_lookup()
    
    # Add USDA code to products
    def get_usda_code(product_code):
        import re
        # Normalize product code
        if isinstance(product_code, str):
            product_code = re.sub(r'-\d+$', '', product_code).strip()
            product_code = re.sub(r'^\d(\d+)$', r'\1', product_code)
        else:
            product_code = str(product_code).strip()
        return usda_lookup.get(product_code, 'NOT_FOUND')
    
    unique_products_df['usda_code'] = unique_products_df['product_code'].apply(get_usda_code)
    
    # Filter based on scope
    if data_scope == 'limited':
        # First filter for only products with valid USDA codes
        valid_usda_df = unique_products_df[unique_products_df['usda_code'] != 'NOT_FOUND'].copy()
        print(f"Found {len(valid_usda_df)} products with valid USDA codes out of {len(unique_products_df)}")
        
        # Then apply the limit if specified
        if limit and limit > 0 and limit < len(valid_usda_df):
            filtered_df = valid_usda_df.sample(limit, random_state=42)
            print(f"Randomly selected {limit} products with valid USDA codes")
        else:
            filtered_df = valid_usda_df
            print(f"Using all {len(filtered_df)} products with valid USDA codes (limit not applied)")
    
    elif data_scope == 'usda_only':
        filtered_df = unique_products_df[unique_products_df['usda_code'] != 'NOT_FOUND'].copy()
        print(f"Selected {len(filtered_df)} products with valid USDA codes out of {len(unique_products_df)}")
    
    else:  # 'all'
        filtered_df = unique_products_df
        print(f"Using all {len(filtered_df)} products")
    
    # 3. Initialize embedder
    print(f"\n3. Initializing {embedding_type} embedder...")
    
    if embedding_type == 'openai':
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key not found in environment")
            return
        
        embedder = OpenAIEmbedder(api_key=api_key)
    else:  # 'local'
        try:
            embedder = LocalEmbedder(model_name=config.SENTENCE_TRANSFORMER_MODEL)
        except Exception as e:
            print(f"Error initializing local embedder: {e}")
            return
    
    # 4. Generate embeddings
    print(f"\n4. Generating embeddings for {len(filtered_df)} products...")
    
    try:
        # Get product descriptions
        descriptions = filtered_df['product_description'].tolist()
        
        # For large datasets, process in batches to show progress
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(descriptions), batch_size):
            end_idx = min(i + batch_size, len(descriptions))
            batch = descriptions[i:end_idx]
            print(f"  Processing batch {i//batch_size + 1}/{(len(descriptions)-1)//batch_size + 1} "
                  f"({i}-{end_idx-1})...")
            
            batch_embeddings = embedder(batch)
            all_embeddings.extend(batch_embeddings)
        
        print(f"Successfully generated {len(all_embeddings)} embeddings")
        
        # Add embeddings to dataframe
        filtered_df['embedding'] = all_embeddings
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return
    
    # 5. Perform sample similarity search
    print("\n5. Performing sample similarity searches...")
    
    # Select a few samples for testing
    test_samples = filtered_df.sample(min(5, len(filtered_df)), random_state=42)
    
    # Track metrics for overall evaluation
    metrics = {
        'total_samples': len(test_samples),
        'correct_top1': 0,
        'correct_top3': 0,
        'correct_top5': 0,
        'avg_rank_of_correct': 0,
        'samples_with_found_correct': 0,
        'mrr': 0  # Mean Reciprocal Rank
    }
    
    for idx, sample in test_samples.iterrows():
        query = sample['product_description']
        query_embedding = sample['embedding']
        query_usda = sample['usda_code']
        
        print(f"\nTest query: '{query}'")
        print(f"USDA code: {query_usda}")
        
        # Calculate similarity to all other products
        similarities = []
        
        for _, row in filtered_df.iterrows():
            if _ == idx:  # Skip self
                continue
            
            # Calculate cosine similarity
            a = query_embedding
            b = row['embedding']
            similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            similarities.append({
                'product_description': row['product_description'],
                'product_code': row['product_code'],
                'usda_code': row['usda_code'],
                'similarity': similarity
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Display top matches
        print(f"Top {k_samples} matches:")
        
        # Check if correct USDA code appears in results
        correct_rank = None
        for i, match in enumerate(similarities):
            # Option 1: Exact match
            exact_match = match['usda_code'] == query_usda
            
            # Option 2: Partial match (if USDA codes share the first part before spaces)
            partial_match = False
            if ' ' in query_usda and ' ' in match['usda_code']:
                query_base = query_usda.split(' ')[0]
                match_base = match['usda_code'].split(' ')[0]
                partial_match = query_base == match_base and len(query_base) >= 3
            
            if exact_match or partial_match:
                correct_rank = i + 1
                break
        
        # Display matches with indication if they match the query USDA code
        for i, match in enumerate(similarities[:k_samples]):
            # Determine match type
            exact_match = match['usda_code'] == query_usda
            
            partial_match = False
            if not exact_match and ' ' in query_usda and ' ' in match['usda_code']:
                query_base = query_usda.split(' ')[0]
                match_base = match['usda_code'].split(' ')[0]
                partial_match = query_base == match_base and len(query_base) >= 3
            
            # Set match marker and label
            if exact_match:
                match_marker = "✅" # Green checkmark
                match_type = "EXACT"
            elif partial_match:
                match_marker = "✅*" # Green checkmark with asterisk for partial
                match_type = "PARTIAL"
            else:
                match_marker = "❌" # Red X
                match_type = "NO MATCH"
                
            print(f"{i+1}. {match_marker} {match['product_description']}")
            print(f"   USDA: {match['usda_code']}, Similarity: {match['similarity']:.4f}, Match: {match_type}")
        
        # Update metrics based on results
        if correct_rank is not None:
            metrics['samples_with_found_correct'] += 1
            metrics['mrr'] += 1.0 / correct_rank
            
            if correct_rank == 1:
                metrics['correct_top1'] += 1
            if correct_rank <= 3:
                metrics['correct_top3'] += 1
            if correct_rank <= 5:
                metrics['correct_top5'] += 1
            
            metrics['avg_rank_of_correct'] += correct_rank
            
            print(f"\nCorrect USDA code found at rank {correct_rank}")
        else:
            print(f"\nCorrect USDA code not found in results")
        
        # 6. Apply LLM verification if requested
        if use_llm_verify:
            try:
                print("\nRunning LLM verification...")
                
                # Prepare candidates for LLM
                candidates = []
                for match in similarities[:k_samples]:
                    candidates.append((match['usda_code'], match['similarity']))
                
                # Direct API call without using the OpenAI client
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment")
                
                # Format candidates for the prompt
                candidates_text = "\n".join([
                    f"{i+1}. {code} (similarity score: {score:.4f})"
                    for i, (code, score) in enumerate(candidates)
                ])
                
                # Create prompt
                prompt = f"""
You are a food product classification expert. Your task is to select the most appropriate USDA code for a given food product description.

PRODUCT DESCRIPTION: {query}

TOP CANDIDATE USDA CODES (with similarity scores from embedding model):
{candidates_text}

Based on your expertise in food products and USDA classification standards, which of these USDA codes is the MOST appropriate match for this product description?

Provide your answer in the following format:
SELECTED CODE: [the selected USDA code]
CONFIDENCE: [score between 0 and 1]
REASONING: [brief explanation of your selection]
"""
                
                import requests
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                
                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a meat product classification expert."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 500
                }
                
                api_url = "https://api.openai.com/v1/chat/completions"
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                response_data = response.json()
                response_text = response_data["choices"][0]["message"]["content"]
                
                # Parse the response
                selected_code_line = [line for line in response_text.split('\n') if line.startswith('SELECTED CODE:')]
                selected_usda = selected_code_line[0].replace('SELECTED CODE:', '').strip() if selected_code_line else None
                
                confidence_line = [line for line in response_text.split('\n') if line.startswith('CONFIDENCE:')]
                confidence = float(confidence_line[0].replace('CONFIDENCE:', '').strip()) if confidence_line else 0.0
                
                reasoning_parts = response_text.split('REASONING:')
                reasoning = reasoning_parts[1].strip() if len(reasoning_parts) > 1 else "No reasoning provided"
                
                print(f"LLM selected USDA code: {selected_usda}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Reasoning: {reasoning}")
                
                # Evaluate against actual USDA code
                correct = selected_usda == query_usda
                match_text = "✅ YES" if correct else "❌ NO"
                print(f"Matches actual USDA code: {match_text}")
                
            except Exception as e:
                print(f"Error in LLM verification: {e}")
                print("Falling back to highest similarity match")
        
        print('-' * 60)
    
    # Calculate final metrics
    if metrics['samples_with_found_correct'] > 0:
        metrics['avg_rank_of_correct'] /= metrics['samples_with_found_correct']
    metrics['mrr'] /= metrics['total_samples']
    
    # Display accuracy summary
    print(f"\n{'='*80}")
    print("ACCURACY SUMMARY:")
    print(f"Total samples evaluated: {metrics['total_samples']}")
    print(f"Correct USDA code in top 1: {metrics['correct_top1']} ({metrics['correct_top1']/metrics['total_samples']*100:.1f}%)")
    print(f"Correct USDA code in top 3: {metrics['correct_top3']} ({metrics['correct_top3']/metrics['total_samples']*100:.1f}%)")
    print(f"Correct USDA code in top 5: {metrics['correct_top5']} ({metrics['correct_top5']/metrics['total_samples']*100:.1f}%)")
    print(f"Mean Reciprocal Rank (MRR): {metrics['mrr']:.4f}")
    
    if metrics['samples_with_found_correct'] > 0:
        print(f"Average rank of correct match (when found): {metrics['avg_rank_of_correct']:.2f}")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    
    # Save results to file if specified
    results_file = f"results_{embedding_type}_{data_scope}.txt"
    print(f"\nResults saved to: {results_file}")
    
    with open(results_file, 'w') as f:
        f.write(f"Analysis Results - {embedding_type.upper()} embedding, {data_scope} scope\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total samples evaluated: {metrics['total_samples']}\n")
        f.write(f"Correct USDA code in top 1: {metrics['correct_top1']} ({metrics['correct_top1']/metrics['total_samples']*100:.1f}%)\n")
        f.write(f"Correct USDA code in top 3: {metrics['correct_top3']} ({metrics['correct_top3']/metrics['total_samples']*100:.1f}%)\n")
        f.write(f"Correct USDA code in top 5: {metrics['correct_top5']} ({metrics['correct_top5']/metrics['total_samples']*100:.1f}%)\n")
        f.write(f"Mean Reciprocal Rank (MRR): {metrics['mrr']:.4f}\n")
        
        if metrics['samples_with_found_correct'] > 0:
            f.write(f"Average rank of correct match (when found): {metrics['avg_rank_of_correct']:.2f}\n")


def get_interactive_options():
    """Get analysis options interactively from user input."""
    # 1. Choose embedding type
    print("Select embedding type:")
    print("1) OpenAI embedding")
    print("2) Local embedding (sentence-transformers)")
    
    while True:
        try:
            embedding_choice = input("\nEnter choice (1-2): ")
            if embedding_choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")
        except Exception:
            print("Please enter a valid number.")
    
    embedding_type = 'openai' if embedding_choice == '1' else 'local'
    
    # 2. Choose data scope
    print("\nSelect data scope:")
    print("1) Limited number of test products (with valid USDA codes)")
    print("2) Only products with matching USDA code")
    print("3) All products")
    
    while True:
        try:
            scope_choice = input("\nEnter choice (1-3): ")
            if scope_choice in ['1', '2', '3']:
                break
            print("Invalid choice. Please enter a number between 1 and 3.")
        except Exception:
            print("Please enter a valid number.")
    
    # Map choice to scope
    scope_map = {'1': 'limited', '2': 'usda_only', '3': 'all'}
    data_scope = scope_map[scope_choice]
    
    # 3. Get limit if applicable
    limit = None
    if scope_choice == '1':
        while True:
            try:
                limit_input = input("\nEnter product limit (e.g., 100): ")
                limit = int(limit_input)
                if limit > 0:
                    break
                print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
    
    # 4. LLM verification option
    print("\nUse AI/LLM verification?")
    print("1) Yes - use GPT to verify matches")
    print("2) No - use embedding similarity only")
    
    while True:
        try:
            llm_choice = input("\nEnter choice (1-2): ")
            if llm_choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")
        except Exception:
            print("Please enter a valid number.")
    
    use_llm = (llm_choice == '1')
    
    # 5. Get K samples if using LLM
    k_samples = 5  # Default
    if use_llm:
        while True:
            try:
                k_input = input("\nEnter K number of samples for LLM (1-10): ")
                k_samples = int(k_input)
                if 1 <= k_samples <= 10:
                    break
                print("Please enter a number between 1 and 10.")
            except ValueError:
                print("Please enter a valid number.")
    
    return embedding_type, data_scope, limit, use_llm, k_samples

def main():
    """Main function to get options from command-line arguments or interactive input."""
    import argparse
    import sys
    
    print("\n=== VectorDB Analysis Tool ===\n")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run analysis on VectorDB product matching')
    
    # Embedding type
    parser.add_argument('--embedding', '-e', type=str, choices=['openai', 'local'],
                        help='Embedding type (openai or local)')
    
    # Data scope
    parser.add_argument('--scope', '-s', type=str, choices=['limited', 'usda_only', 'all'],
                        help='Data scope (limited, usda_only, or all)')
    
    # Product limit
    parser.add_argument('--limit', '-l', type=int,
                        help='Maximum number of products to process')
    
    # LLM verification
    parser.add_argument('--llm', action='store_true',
                        help='Use LLM verification')
    
    # K samples for LLM
    parser.add_argument('--k-samples', '-k', type=int,
                        help='Number of samples to use for LLM verification')
    
    # Interactive mode flag
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode with prompts')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine if we should use interactive mode
    use_interactive = args.interactive or (
        args.embedding is None and 
        args.scope is None and 
        args.limit is None and 
        not args.llm and 
        args.k_samples is None and
        len(sys.argv) == 1  # Only the script name was provided
    )
    
    # Get options either interactively or from command line
    if use_interactive:
        print("Running in interactive mode...")
        embedding_type, data_scope, limit, use_llm, k_samples = get_interactive_options()
    else:
        # Use command line arguments
        embedding_type = args.embedding or 'openai'
        data_scope = args.scope or 'limited'
        limit = args.limit if args.limit is not None else 100
        use_llm = args.llm
        k_samples = args.k_samples if args.k_samples is not None else 5
    
    # Print summary of options
    print("\nSelected options:")
    print(f"  - Embedding type: {embedding_type}")
    print(f"  - Data scope: {data_scope}")
    if data_scope == 'limited':
        print(f"  - Product limit: {limit}")
    print(f"  - LLM verification: {'Yes' if use_llm else 'No'}")
    if use_llm:
        print(f"  - K samples: {k_samples}")
    
    # Run the analysis with selected options
    print("\nStarting analysis with selected options...")
    run_analysis(
        embedding_type=embedding_type,
        data_scope=data_scope,
        limit=limit if data_scope == 'limited' else None,
        use_llm_verify=use_llm,
        k_samples=k_samples
    )

if __name__ == "__main__":
    main()
