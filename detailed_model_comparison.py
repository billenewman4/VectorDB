#!/usr/bin/env python3
"""
Detailed Model Comparison

Analyzes each test case to understand the exact behavior of the embedding model
and cross-encoder, allowing us to diagnose why the weighted approach isn't 
showing improved results.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import time
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.VectorDB.CrossEncoder import CrossEncoder
try:
    from src.VectorDB.localEmbedder import LocalEmbedder
except ImportError:
    print("Warning: LocalEmbedder import failed")

try:
    from src.VectorDB.helper import build_usda_lookup, preprocess_text_for_matching
except ImportError:
    print("Warning: helper module import failed")

try:
    import src.config as config
except ImportError:
    print("Warning: config module import failed")

def get_usda_lookup():
    """
    Get the USDA code lookup map
    """
    try:
        # Use the build_usda_lookup function from helper.py
        lookup_map = build_usda_lookup(
            mapping_file=config.GROUND_TRUTH_FILE,
            sheet_name=config.GROUND_TRUTH_SHEET_NAME,
            id_cols=config.GROUND_TRUTH_ID_COLS,
            usda_col=config.GROUND_TRUTH_USDA_COL
        )
        return lookup_map
    except Exception as e:
        print(f"Error building USDA lookup: {e}")
        return {}

def get_usda_codes():
    """
    Get a list of unique USDA codes from the lookup map
    """
    usda_lookup = get_usda_lookup()
    unique_codes = set(usda_lookup.values())
    return list(unique_codes)

def get_sample_products():
    """
    Get the same sample products that would be used in run_analysis.py
    Try to match the exact 20 samples being tested
    """
    results_file = "results_local_usda_only.txt"
    try:
        # Try to extract detailed results from the file if it contains them
        with open(results_file, 'r') as f:
            content = f.read()
            
            # Look for sample information
            import re
            samples = re.findall(r"Sample \d+: '([^']+)' → Correct USDA: '([^']+)'", content)
            
            if samples:
                print(f"Found {len(samples)} samples in results file")
                return [{'description': s[0], 'correct_usda': s[1]} for s in samples]
    except Exception as e:
        print(f"Error reading results file: {e}")
    
    print("Could not extract sample data from results file. Loading from transaction data...")
    
    # If we can't find the samples in the results file, try to load transaction data
    try:
        from src.data_processing import load_transaction_data, process_transaction_data
        
        raw_data = load_transaction_data()
        if raw_data is None:
            print("Error: Failed to load transaction data")
            return []
        
        unique_products_df = process_transaction_data(raw_data)
        if unique_products_df is None or len(unique_products_df) == 0:
            print("Error: No products found after processing")
            return []
        
        # Add USDA code to products
        usda_lookup = get_usda_lookup()
        
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
        
        # Filter for products with valid USDA codes
        valid_usda_df = unique_products_df[unique_products_df['usda_code'] != 'NOT_FOUND'].copy()
        print(f"Found {len(valid_usda_df)} products with valid USDA codes")
        
        # Sample 20 products with valid USDA codes
        if len(valid_usda_df) > 20:
            sampled_df = valid_usda_df.sample(20, random_state=42)  # Use the same random seed as run_analysis.py
        else:
            sampled_df = valid_usda_df
        
        # Format the samples
        samples = []
        for _, row in sampled_df.iterrows():
            samples.append({
                'description': row['product_description'],
                'correct_usda': row['usda_code']
            })
        
        print(f"Sampled {len(samples)} products for analysis")
        return samples
        
    except Exception as e:
        print(f"Error loading transaction data: {e}")
        
    # If all else fails, create a few sample products manually
    print("Warning: Creating manual sample products. These may not match the ones in the results file.")
    return [
        {'description': 'beef ribeye lip-on', 'correct_usda': '112a HVY Lipon Ribeye'},
        {'description': 'beef short ribs bone-in', 'correct_usda': '123 A 3 short rib choice'},
        {'description': 'beef chuck pectoral meat choice', 'correct_usda': 'Pectoral'},
        {'description': 'beef skirt outer', 'correct_usda': '121 C 4 Outside skirt'},
        {'description': 'beef rib meat lifter', 'correct_usda': 'Cap & Wedge'}
    ]

def analyze_models():
    """
    Analyze the performance of the embedding model and cross-encoder
    on the same set of sample products
    """
    print("\n" + "="*80)
    print("Detailed Model Comparison")
    print("="*80)
    
    # Initialize models
    print("Initializing models...")
    cross_encoder = CrossEncoder()
    print(f"Cross-encoder weights: CE={cross_encoder.cross_encoder_weight:.1f}, Emb={cross_encoder.embedding_weight:.1f}")
    
    try:
        embedder = LocalEmbedder()
        print(f"Local embedder model: {embedder.model_name}")
        embedding_available = True
    except Exception as e:
        print(f"Error initializing LocalEmbedder: {e}")
        print("Will proceed without embedding comparison")
        embedding_available = False
        embedder = None
    
    # Get all USDA codes to test against
    all_usda_codes = get_usda_codes()
    if not all_usda_codes:
        print("Error: No USDA codes found")
        return
    
    print(f"Found {len(all_usda_codes)} unique USDA codes to test against")
    
    # Get sample products
    samples = get_sample_products()
    if not samples:
        print("Error: No sample products found")
        return
    
    print(f"Analyzing {len(samples)} sample products...")
    
    # Store detailed results for each sample
    detailed_results = []
    summary_results = []
    
    # Process each sample
    for i, sample in enumerate(samples):
        print(f"\n{'-'*80}")
        product = sample['description']
        correct_usda = sample['correct_usda']
        
        print(f"Sample {i+1}: '{product}' → Correct USDA: '{correct_usda}'")
        
        # Track if the correct USDA code is in our list of codes to test
        correct_code_found = correct_usda in all_usda_codes
        if not correct_code_found:
            print(f"WARNING: Correct USDA code '{correct_usda}' not found in our list of USDA codes!")
            # Add it to the list
            all_usda_codes.append(correct_usda)
        
        if embedding_available:
            # Generate embedding for product
            try:
                product_embedding = embedder([product])[0]
                usda_embeddings = embedder(all_usda_codes)
                
                # Calculate embedding similarities
                embedding_scores = []
                for j, usda_embedding in enumerate(usda_embeddings):
                    from sklearn.metrics.pairwise import cosine_similarity
                    product_embedding_2d = np.array(product_embedding).reshape(1, -1)
                    usda_embedding_2d = np.array(usda_embedding).reshape(1, -1)
                    
                    similarity = float(cosine_similarity(product_embedding_2d, usda_embedding_2d)[0][0])
                    embedding_scores.append({
                        'usda_code': all_usda_codes[j],
                        'similarity': similarity,
                        'is_correct': all_usda_codes[j] == correct_usda
                    })
                
                # Sort by embedding similarity
                embedding_sorted = sorted(embedding_scores, key=lambda x: x['similarity'], reverse=True)
                
                # Find rank of correct USDA code
                embedding_rank = None
                for j, result in enumerate(embedding_sorted):
                    if result['is_correct']:
                        embedding_rank = j + 1
                        break
                
                print(f"Embedding rank of correct USDA: {embedding_rank} / {len(all_usda_codes)}")
                print("\nTop 5 matches by embedding similarity:")
                for j, result in enumerate(embedding_sorted[:5]):
                    indicator = " [CORRECT]" if result['is_correct'] else ""
                    print(f"  {j+1}. {result['usda_code']}{indicator} - Score: {result['similarity']:.4f}")
                
            except Exception as e:
                print(f"Error calculating embedding similarities: {e}")
                embedding_rank = None
                embedding_sorted = []
        else:
            embedding_rank = None
            embedding_sorted = []
        
        # Create candidate dictionaries for cross-encoder ranking
        candidate_dicts = []
        for usda_code in all_usda_codes:
            if embedding_available and any(e['usda_code'] == usda_code for e in embedding_sorted):
                # Use the actual embedding similarity if available
                similarity = next(e['similarity'] for e in embedding_sorted if e['usda_code'] == usda_code)
            else:
                # Use a placeholder similarity
                similarity = 0.5
            
            candidate_dicts.append({
                'usda_code': usda_code,
                'similarity': similarity,
                'is_correct': usda_code == correct_usda
            })
        
        # Run pure cross-encoder scoring (without weighted combination)
        pure_cross_encoder_scores = []
        text_pairs = [[product, candidate['usda_code']] for candidate in candidate_dicts]
        raw_scores = cross_encoder.model.predict(text_pairs)
        
        for j, score in enumerate(raw_scores):
            pure_cross_encoder_scores.append({
                'usda_code': candidate_dicts[j]['usda_code'],
                'similarity': float(score),
                'is_correct': candidate_dicts[j]['is_correct']
            })
        
        # Sort by pure cross-encoder scores
        pure_sorted = sorted(pure_cross_encoder_scores, key=lambda x: x['similarity'], reverse=True)
        
        # Find rank of correct USDA code
        pure_ce_rank = None
        for j, result in enumerate(pure_sorted):
            if result['is_correct']:
                pure_ce_rank = j + 1
                break
        
        print(f"\nPure Cross-Encoder rank of correct USDA: {pure_ce_rank} / {len(all_usda_codes)}")
        print("\nTop 5 matches by pure cross-encoder scores:")
        for j, result in enumerate(pure_sorted[:5]):
            indicator = " [CORRECT]" if result['is_correct'] else ""
            print(f"  {j+1}. {result['usda_code']}{indicator} - Score: {result['similarity']:.4f}")
        
        # Run weighted cross-encoder (the approach we're now using)
        # Note: This uses the candidate_dicts which already have embedding similarities
        weighted_results = cross_encoder.rerank(product, candidate_dicts)
        
        # Find rank of correct USDA code
        weighted_rank = None
        for j, result in enumerate(weighted_results):
            if result['usda_code'] == correct_usda:
                weighted_rank = j + 1
                break
        
        print(f"\nWeighted Cross-Encoder rank of correct USDA: {weighted_rank} / {len(all_usda_codes)}")
        print("\nTop 5 matches by weighted scores:")
        for j, result in enumerate(weighted_results[:5]):
            indicator = " [CORRECT]" if result['usda_code'] == correct_usda else ""
            print(f"  {j+1}. {result['usda_code']}{indicator} - Score: {result['similarity']:.4f}")
            if 'embedding_score' in result and 'cross_encoder_score' in result:
                print(f"     Embedding: {result['embedding_score']:.4f}, Cross-Encoder: {result['cross_encoder_score']:.4f}")
        
        # Store detailed results for this sample
        sample_result = {
            'sample_id': i+1,
            'product_description': product,
            'correct_usda': correct_usda,
            'embedding_rank': embedding_rank,
            'pure_cross_encoder_rank': pure_ce_rank,
            'weighted_rank': weighted_rank,
            'embedding_top_matches': embedding_sorted[:5] if embedding_available else [],
            'pure_ce_top_matches': pure_sorted[:5],
            'weighted_top_matches': weighted_results[:5]
        }
        detailed_results.append(sample_result)
        
        # Store summary for this sample
        summary_results.append({
            'sample_id': i+1,
            'product_description': product,
            'correct_usda': correct_usda,
            'embedding_rank': embedding_rank,
            'pure_cross_encoder_rank': pure_ce_rank,
            'weighted_rank': weighted_rank,
            'embedding_improved': embedding_rank is not None and weighted_rank is not None and weighted_rank < embedding_rank,
            'cross_encoder_improved': pure_ce_rank is not None and weighted_rank is not None and weighted_rank < pure_ce_rank,
            'embedding_maintained': embedding_rank is not None and weighted_rank is not None and weighted_rank == embedding_rank and weighted_rank <= 3,
            'cross_encoder_maintained': pure_ce_rank is not None and weighted_rank is not None and weighted_rank == pure_ce_rank and weighted_rank <= 3,
            'embedding_degraded': embedding_rank is not None and weighted_rank is not None and weighted_rank > embedding_rank,
            'cross_encoder_degraded': pure_ce_rank is not None and weighted_rank is not None and weighted_rank > pure_ce_rank,
        })
    
    # Print summary of results
    print("\n" + "="*80)
    print("Summary of Results")
    print("="*80)
    
    total_samples = len(summary_results)
    embedding_top1 = sum(1 for r in summary_results if r['embedding_rank'] == 1)
    pure_ce_top1 = sum(1 for r in summary_results if r['pure_cross_encoder_rank'] == 1)
    weighted_top1 = sum(1 for r in summary_results if r['weighted_rank'] == 1)
    
    embedding_top3 = sum(1 for r in summary_results if r['embedding_rank'] is not None and r['embedding_rank'] <= 3)
    pure_ce_top3 = sum(1 for r in summary_results if r['pure_cross_encoder_rank'] is not None and r['pure_cross_encoder_rank'] <= 3)
    weighted_top3 = sum(1 for r in summary_results if r['weighted_rank'] is not None and r['weighted_rank'] <= 3)
    
    print(f"Total samples analyzed: {total_samples}")
    
    print("\nEmbedding model performance:")
    print(f"  Top-1 accuracy: {embedding_top1} / {total_samples} ({embedding_top1/total_samples*100:.1f}%)")
    print(f"  Top-3 accuracy: {embedding_top3} / {total_samples} ({embedding_top3/total_samples*100:.1f}%)")
    
    print("\nPure Cross-Encoder performance:")
    print(f"  Top-1 accuracy: {pure_ce_top1} / {total_samples} ({pure_ce_top1/total_samples*100:.1f}%)")
    print(f"  Top-3 accuracy: {pure_ce_top3} / {total_samples} ({pure_ce_top3/total_samples*100:.1f}%)")
    
    print("\nWeighted Cross-Encoder performance:")
    print(f"  Top-1 accuracy: {weighted_top1} / {total_samples} ({weighted_top1/total_samples*100:.1f}%)")
    print(f"  Top-3 accuracy: {weighted_top3} / {total_samples} ({weighted_top3/total_samples*100:.1f}%)")
    
    # Count improvements and degradations vs embedding
    vs_embedding_improved = sum(1 for r in summary_results if r['embedding_improved'])
    vs_embedding_maintained = sum(1 for r in summary_results if r['embedding_maintained'])
    vs_embedding_degraded = sum(1 for r in summary_results if r['embedding_degraded'])
    
    # Count improvements and degradations vs pure cross-encoder
    vs_ce_improved = sum(1 for r in summary_results if r['cross_encoder_improved'])
    vs_ce_maintained = sum(1 for r in summary_results if r['cross_encoder_maintained'])
    vs_ce_degraded = sum(1 for r in summary_results if r['cross_encoder_degraded'])
    
    print("\nWeighted vs Embedding Model:")
    print(f"  Improved: {vs_embedding_improved} / {total_samples} ({vs_embedding_improved/total_samples*100:.1f}%)")
    print(f"  Maintained good rank: {vs_embedding_maintained} / {total_samples} ({vs_embedding_maintained/total_samples*100:.1f}%)")
    print(f"  Degraded: {vs_embedding_degraded} / {total_samples} ({vs_embedding_degraded/total_samples*100:.1f}%)")
    
    print("\nWeighted vs Pure Cross-Encoder:")
    print(f"  Improved: {vs_ce_improved} / {total_samples} ({vs_ce_improved/total_samples*100:.1f}%)")
    print(f"  Maintained good rank: {vs_ce_maintained} / {total_samples} ({vs_ce_maintained/total_samples*100:.1f}%)")
    print(f"  Degraded: {vs_ce_degraded} / {total_samples} ({vs_ce_degraded/total_samples*100:.1f}%)")
    
    # Save detailed results to CSV
    detailed_df = pd.DataFrame(summary_results)
    detailed_df.to_csv("detailed_model_comparison.csv", index=False)
    print("\nDetailed results saved to 'detailed_model_comparison.csv'")
    
    # Plot comparison of ranks
    plt.figure(figsize=(12, 8))
    x = range(1, total_samples+1)
    
    # Extract ranks and replace None with a high value for plotting
    high_value = len(all_usda_codes) + 1  # Higher than any possible rank
    embedding_ranks = [r['embedding_rank'] if r['embedding_rank'] is not None else high_value for r in summary_results]
    pure_ce_ranks = [r['pure_cross_encoder_rank'] if r['pure_cross_encoder_rank'] is not None else high_value for r in summary_results]
    weighted_ranks = [r['weighted_rank'] if r['weighted_rank'] is not None else high_value for r in summary_results]
    
    plt.plot(x, embedding_ranks, 'o-', label="Embedding")
    plt.plot(x, pure_ce_ranks, 's-', label="Pure Cross-Encoder")
    plt.plot(x, weighted_ranks, '^-', label="Weighted")
    
    plt.xlabel("Sample ID")
    plt.ylabel("Rank of Correct USDA Code")
    plt.title("Comparison of Ranks Across Models")
    plt.yscale('log')  # Use log scale to better show differences in ranks
    plt.grid(True)
    plt.legend()
    
    # Add a horizontal line at rank 3
    plt.axhline(y=3, color='r', linestyle='--', alpha=0.5, label="Top-3 Threshold")
    
    plt.savefig("rank_comparison.png")
    print("Rank comparison visualization saved to 'rank_comparison.png'")
    
    # Determine if there's a problem with the run_analysis.py integration
    print("\nDiagnosing run_analysis.py integration:")
    
    if weighted_top1 == pure_ce_top1 and weighted_top3 == pure_ce_top3:
        print("ISSUE DETECTED: Weighted approach is showing identical performance to pure cross-encoder.")
        print("This suggests the weighted approach may not be properly being used in run_analysis.py.")
        
        # Check if the weights might be set to ignore embedding scores
        if cross_encoder.cross_encoder_weight == 1.0 and cross_encoder.embedding_weight == 0.0:
            print("The weights are set to completely ignore embedding scores (CE=1.0, Emb=0.0).")
            print("Recommendation: Update the CrossEncoder initialization to use balanced weights.")
        else:
            print(f"The weights are set to CE={cross_encoder.cross_encoder_weight:.1f}, Emb={cross_encoder.embedding_weight:.1f}.")
            print("Recommendation: Check if run_analysis.py is using an old version of CrossEncoder or overriding the weights.")
    
    return detailed_results

if __name__ == "__main__":
    results = analyze_models()
