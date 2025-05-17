"""
USDA Mapping Summary Statistics

This script analyzes the USDA mapping results and generates detailed statistics
on the accuracy and performance of the matching process.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re
from collections import Counter

# Ensure we can import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.abbreviation_translator import expand_abbreviations


def load_results(results_file):
    """Load the USDA matching results file."""
    print(f"Loading results from: {results_file}")
    
    try:
        if results_file.endswith('.xlsx'):
            df = pd.read_excel(results_file)
        else:
            df = pd.read_csv(results_file)
        
        print(f"Loaded {len(df)} product records.")
        return df
    except Exception as e:
        print(f"Error loading results file: {e}")
        return None


def filter_known_usda_products(df):
    """Filter the dataframe to only include products with known USDA mappings."""
    known_df = df.dropna(subset=['known_usda_code'])
    print(f"Found {len(known_df)} products with known USDA mappings out of {len(df)} total products.")
    return known_df


def calculate_similarity_statistics(df):
    """Calculate statistics about similarity scores."""
    similarity_stats = {
        'mean': df['similarity_score'].mean(),
        'median': df['similarity_score'].median(),
        'min': df['similarity_score'].min(),
        'max': df['similarity_score'].max(),
        'std': df['similarity_score'].std()
    }
    
    print("\n--- Similarity Score Statistics ---")
    print(f"Mean similarity: {similarity_stats['mean']:.4f}")
    print(f"Median similarity: {similarity_stats['median']:.4f}")
    print(f"Min similarity: {similarity_stats['min']:.4f}")
    print(f"Max similarity: {similarity_stats['max']:.4f}")
    print(f"Standard deviation: {similarity_stats['std']:.4f}")
    
    return similarity_stats


def analyze_matches(df):
    """Analyze the matching results and calculate accuracy metrics."""
    # Basic accuracy
    correct_matches = df[df['is_correct_match'] == True].shape[0]
    total_products = len(df)
    accuracy = correct_matches / total_products if total_products > 0 else 0
    
    print("\n--- Matching Accuracy ---")
    print(f"Correctly matched USDA codes: {correct_matches} out of {total_products} ({accuracy:.2%})")
    
    # Similarity score difference between correct and incorrect matches
    correct_sim = df[df['is_correct_match'] == True]['similarity_score'].mean()
    incorrect_sim = df[df['is_correct_match'] == False]['similarity_score'].mean()
    
    print(f"Average similarity for correct matches: {correct_sim:.4f}")
    print(f"Average similarity for incorrect matches: {incorrect_sim:.4f}")
    print(f"Difference: {correct_sim - incorrect_sim:.4f}")
    
    return {
        'accuracy': accuracy,
        'correct_matches': correct_matches,
        'total_products': total_products,
        'correct_sim_avg': correct_sim,
        'incorrect_sim_avg': incorrect_sim
    }


def analyze_usda_code_performance(df):
    """Analyze which USDA codes are most frequently matched correctly/incorrectly."""
    # Count occurrences of each known USDA code
    known_code_counts = Counter(df['known_usda_code'])
    
    # For each known USDA code, calculate how often it was matched correctly
    code_performance = {}
    for code, count in known_code_counts.items():
        code_df = df[df['known_usda_code'] == code]
        correct = code_df[code_df['is_correct_match'] == True].shape[0]
        code_performance[code] = {
            'total': count,
            'correct': correct,
            'accuracy': correct / count if count > 0 else 0
        }
    
    # Sort by frequency
    sorted_codes = sorted(code_performance.items(), key=lambda x: x[1]['total'], reverse=True)
    
    print("\n--- Performance by USDA Code ---")
    print(f"{'USDA Code':<30} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    print('-' * 60)
    
    for code, stats in sorted_codes[:10]:  # Show top 10
        print(f"{code:<30} {stats['total']:<8} {stats['correct']:<8} {stats['accuracy']:.2%}")
    
    return code_performance


def analyze_common_mismatches(df):
    """Analyze the most common incorrect mappings."""
    incorrect_df = df[df['is_correct_match'] == False]
    
    # Create pairs of (known_code, predicted_code)
    mismatch_pairs = list(zip(incorrect_df['known_usda_code'], incorrect_df['best_matching_usda_code']))
    
    # Count occurrences of each mismatch pair
    mismatch_counts = Counter(mismatch_pairs)
    
    # Sort by frequency
    sorted_mismatches = sorted(mismatch_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n--- Most Common Mismatches ---")
    print(f"{'Known USDA Code':<30} {'Predicted USDA Code':<30} {'Count':<8}")
    print('-' * 70)
    
    for (known, predicted), count in sorted_mismatches[:10]:  # Show top 10
        print(f"{known:<30} {predicted:<30} {count:<8}")
    
    return mismatch_counts


def analyze_similarity_threshold(df):
    """Analyze how accuracy changes with different similarity thresholds."""
    thresholds = np.arange(0.1, 1.0, 0.05)
    accuracies = []
    
    for threshold in thresholds:
        # Only consider matches above threshold
        filtered_df = df[df['similarity_score'] >= threshold]
        if len(filtered_df) > 0:
            correct = filtered_df[filtered_df['is_correct_match'] == True].shape[0]
            accuracy = correct / len(filtered_df)
        else:
            accuracy = 0
        
        accuracies.append(accuracy)
        
    results = pd.DataFrame({
        'threshold': thresholds,
        'accuracy': accuracies,
        'remaining_samples': [len(df[df['similarity_score'] >= t]) for t in thresholds]
    })
    
    print("\n--- Accuracy by Similarity Threshold ---")
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Remaining Samples':<20} {'Percentage Remaining':<20}")
    print('-' * 60)
    
    total_samples = len(df)
    for _, row in results.iterrows():
        remaining_pct = row['remaining_samples'] / total_samples if total_samples > 0 else 0
        print(f"{row['threshold']:<10.2f} {row['accuracy']:<10.2%} {row['remaining_samples']:<20} {remaining_pct:<20.2%}")
    
    return results


def analyze_product_descriptions(df):
    """Analyze patterns in product descriptions for correct vs incorrect matches."""
    # Get word frequencies for correct and incorrect matches
    correct_df = df[df['is_correct_match'] == True]
    incorrect_df = df[df['is_correct_match'] == False]
    
    # Sample a few correct matches for analysis
    print("\n--- Sample Correct Matches ---")
    print(f"{'Product Description':<50} {'Known USDA Code':<30} {'Predicted USDA Code':<30} {'Similarity':<10}")
    print('-' * 120)
    
    for _, row in correct_df.sample(min(5, len(correct_df))).iterrows():
        print(f"{row['product_description'][:50]:<50} {row['known_usda_code']:<30} {row['best_matching_usda_code']:<30} {row['similarity_score']:.4f}")
    
    # Sample a few incorrect matches for analysis
    print("\n--- Sample Incorrect Matches ---")
    print(f"{'Product Description':<50} {'Known USDA Code':<30} {'Predicted USDA Code':<30} {'Similarity':<10}")
    print('-' * 120)
    
    for _, row in incorrect_df.sample(min(5, len(incorrect_df))).iterrows():
        print(f"{row['product_description'][:50]:<50} {row['known_usda_code']:<30} {row['best_matching_usda_code']:<30} {row['similarity_score']:.4f}")


def generate_detailed_report(stats_dict, output_file):
    """Generate a detailed report file with all statistics."""
    with open(output_file, 'w') as f:
        f.write("# USDA Mapping Summary Statistics Report\n\n")
        
        f.write("## Overall Accuracy\n")
        f.write(f"Correctly matched USDA codes: {stats_dict['match_stats']['correct_matches']} out of "
                f"{stats_dict['match_stats']['total_products']} "
                f"({stats_dict['match_stats']['accuracy']:.2%})\n\n")
        
        f.write("## Similarity Score Statistics\n")
        f.write(f"Mean similarity: {stats_dict['similarity_stats']['mean']:.4f}\n")
        f.write(f"Median similarity: {stats_dict['similarity_stats']['median']:.4f}\n")
        f.write(f"Min similarity: {stats_dict['similarity_stats']['min']:.4f}\n")
        f.write(f"Max similarity: {stats_dict['similarity_stats']['max']:.4f}\n")
        f.write(f"Standard deviation: {stats_dict['similarity_stats']['std']:.4f}\n\n")
        
        f.write("## Average Similarity for Correct vs Incorrect Matches\n")
        f.write(f"Average similarity for correct matches: {stats_dict['match_stats']['correct_sim_avg']:.4f}\n")
        f.write(f"Average similarity for incorrect matches: {stats_dict['match_stats']['incorrect_sim_avg']:.4f}\n")
        f.write(f"Difference: {stats_dict['match_stats']['correct_sim_avg'] - stats_dict['match_stats']['incorrect_sim_avg']:.4f}\n\n")
        
        # Add more sections as needed
        
        f.write("\n\nReport generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    print(f"\nDetailed report saved to: {output_file}")


def main():
    """Main function to run the analysis."""
    results_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "analysis_results", "best_usda_matches.csv")
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return
    
    # Load results
    df = load_results(results_file)
    if df is None:
        return
    
    # Filter to only include products with known USDA mappings
    known_df = filter_known_usda_products(df)
    
    # Run analyses
    similarity_stats = calculate_similarity_statistics(known_df)
    match_stats = analyze_matches(known_df)
    code_performance = analyze_usda_code_performance(known_df)
    mismatch_counts = analyze_common_mismatches(known_df)
    threshold_results = analyze_similarity_threshold(known_df)
    analyze_product_descriptions(known_df)
    
    # Generate detailed report
    report_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              "analysis_results", "usda_mapping_report.md")
    
    stats_dict = {
        'similarity_stats': similarity_stats,
        'match_stats': match_stats,
        'code_performance': code_performance,
        'mismatch_counts': mismatch_counts,
        'threshold_results': threshold_results
    }
    
    generate_detailed_report(stats_dict, report_file)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
