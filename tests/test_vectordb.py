#!/usr/bin/env python3
"""
Test script for the vector database implementation
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import ast
from typing import List, Dict, Tuple, Any, Set

# Add parent directory to path so we can import from src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import functions from vectordb.py
from src.vectordb import (
    ProductEmbedder,
    ProductVectorDB,
    create_product_vector_db,
    find_similar_products
)

# Import functions from excel.py
from src.excel import load_mapping_data, process_transaction_data

def test_embedding_model():
    """Test the embedding model by creating sample embeddings"""
    print("\n=== Testing Embedding Model ===")
    
    # Initialize embedder with default model
    embedder = ProductEmbedder()
    
    # Create test embeddings
    test_texts = [
        "almond milk",
        "frozen beef",
        "fresh vegetables",
        "organic bananas"
    ]
    
    print(f"Creating embeddings for {len(test_texts)} test texts")
    
    for text in test_texts:
        # Create embedding
        embedding = embedder.embed_query(text)
        
        # Check embedding shape and values
        print(f"Text: '{text}'")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"  First 5 values: {embedding[:5]}")
    
    print("✅ Embedding model test completed")
    return embedder

def test_enhanced_descriptions(embedder):
    """Test the creation of enhanced descriptions"""
    print("\n=== Testing Enhanced Descriptions ===")
    
    # Create sample product data
    sample_product = pd.Series({
        'product_description': 'organic milk',
        'avg_price': 4.99,
        'avg_quantity': 32.5
    })
    
    # Create enhanced description
    enhanced_desc = embedder.create_text_description(sample_product)
    print(f"Original: '{sample_product['product_description']}'")
    print(f"Enhanced: '{enhanced_desc}'")
    
    # Check if metrics are included in description
    assert 'average price' in enhanced_desc, "Price not included in enhanced description"
    assert 'average quantity' in enhanced_desc, "Quantity not included in enhanced description"
    
    print("✅ Enhanced descriptions test completed")

def test_vector_db_creation():
    """Test the creation of the vector database"""
    print("\n=== Testing Vector Database Creation ===")
    
    # Record start time
    start_time = time.time()
    
    # Create vector database (with recreate flag)
    print("Creating vector database...")
    vector_db = create_product_vector_db(recreate=True)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Vector database created in {elapsed_time:.2f} seconds")
    
    # Get collection info
    collection = vector_db.client.get_collection("products")
    
    # Get collection count
    count = collection.count()
    print(f"Collection contains {count} items")
    
    # Make sure we have reasonable number of records
    assert count > 0, "Vector database should contain items"
    
    print("✅ Vector database creation test completed")
    return vector_db

def test_similarity_search(vector_db):
    """Test similarity search functionality"""
    print("\n=== Testing Similarity Search ===")
    
    # Define test queries with expected product types
    test_queries = [
        ("almond milk unsweetened", "almond milk"),
        ("premium frozen beef", "beef frozen"),
        ("fresh organic apples", "apple"),
        ("orange juice", "juice"),
    ]
    
    for query, expected_type in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected to find products like: '{expected_type}'")
        
        # Search for similar products
        results = find_similar_products(query, n_results=3, vector_db=vector_db)
        
        # Display results
        print("Top results:")
        for i, (_, row) in enumerate(results.iterrows()):
            print(f"  {i+1}. '{row['product_description']}' (similarity: {row['similarity']:.4f})")
        
        # Check if any result contains the expected type (simple substring check)
        found = any(expected_type.lower() in desc.lower() for desc in results['product_description'])
        print(f"Found expected product type: {'✅' if found else '❌'}")
    
    print("✅ Similarity search test completed")

def test_performance_metrics(vector_db):
    """Test performance metrics for the vector database"""
    print("\n=== Testing Performance Metrics ===")
    
    # Load sample of transaction data for test queries
    _, unique_products = process_transaction_data()
    
    # Use a sample of product descriptions as queries
    sample_size = min(10, len(unique_products))
    test_products = unique_products.sample(sample_size)
    
    # Set up timing statistics
    query_times = []
    batch_size = 3  # Number of similar products to retrieve
    
    print(f"Running {sample_size} sample queries with batch size {batch_size}")
    
    for _, row in test_products.iterrows():
        query = row['product_description']
        
        # Time the query
        start_time = time.time()
        results = vector_db.get_similar_products(query, n_results=batch_size)
        query_time = time.time() - start_time
        
        query_times.append(query_time)
        
        # Check if query found itself as the top result
        found_self = False
        if not results.empty and 'product_description' in results.columns:
            top_result = results.iloc[0]['product_description']
            found_self = query.lower() == top_result.lower()
        
        print(f"Query: '{query[:30]}...' took {query_time:.4f}s, found self: {'✅' if found_self else '❌'}")
    
    # Calculate statistics
    avg_time = sum(query_times) / len(query_times)
    max_time = max(query_times)
    min_time = min(query_times)
    
    print(f"\nPerformance statistics:")
    print(f"  Average query time: {avg_time:.4f}s")
    print(f"  Minimum query time: {min_time:.4f}s")
    print(f"  Maximum query time: {max_time:.4f}s")
    
    print("✅ Performance metrics test completed")

def test_mapping_recall(vector_db):
    """Test recall using the ground truth from the product mapping file"""
    print("\n=== Testing Recall with Product Mapping Ground Truth ===")
    
    # Load the correct mapping data
    mapping_path = Path("CorrectMapping/product_mapping_semantic.xlsx")
    if not mapping_path.exists():
        print(f"❌ Mapping file not found at {mapping_path}")
        return
    
    try:
        # Read the mapping file
        mapping_df = pd.read_excel(mapping_path)
        print(f"Loaded mapping file with {len(mapping_df)} product clusters")
        
        # Metrics to track
        total_queries = 0
        total_possible_matches = 0
        total_found_matches = 0
        recall_by_product = {}
        similarity_threshold = 0.25  # Customizable threshold
        
        # Process each product cluster
        for _, row in mapping_df.iterrows():
            standard_name = row["Standardized Product Name"]
            
            # Get possible descriptions - parse the string into a list
            possible_descriptions_text = row["Possible Descriptions"]
            possible_descriptions = [desc.strip() for desc in possible_descriptions_text.split(";")]
            
            print(f"\nProduct Cluster: '{standard_name}'")
            print(f"  {len(possible_descriptions)} possible variations")
            
            # Track matches found for this cluster
            total_found_in_cluster = 0
            
            # Use standard name as query
            query = standard_name
            print(f"  Query: '{query}'")
            
            # Find similar products with threshold
            results = find_similar_products(query, n_results=20, 
                                           similarity_threshold=similarity_threshold,
                                           vector_db=vector_db)
            
            # Check how many of the possible descriptions were found
            found_descriptions = set()
            if not results.empty:
                # For each result, check if it's in the possible descriptions
                for _, result_row in results.iterrows():
                    product_desc = result_row["product_description"]
                    similarity = result_row["similarity"]
                    
                    # Check for exact matches first
                    match_found = False
                    for possible in possible_descriptions:
                        if product_desc.lower() == possible.lower():
                            match_found = True
                            found_descriptions.add(possible)
                            break
                    
                    match_indicator = "✓" if match_found else "✗"
                    print(f"    {match_indicator} {product_desc} (sim: {similarity:.4f})")
            
            # Update metrics
            found_count = len(found_descriptions)
            total_found_in_cluster = found_count
            cluster_recall = found_count / len(possible_descriptions) if possible_descriptions else 0
            
            print(f"  Found {found_count} out of {len(possible_descriptions)} variations")
            print(f"  Cluster recall: {cluster_recall:.2f}")
            
            # Update global metrics
            total_queries += 1
            total_possible_matches += len(possible_descriptions)
            total_found_matches += total_found_in_cluster
            recall_by_product[standard_name] = cluster_recall
        
        # Calculate overall recall
        overall_recall = total_found_matches / total_possible_matches if total_possible_matches > 0 else 0
        
        # Print summary
        print("\n=== Recall Evaluation Summary ===")
        print(f"Overall recall: {overall_recall:.4f} ({total_found_matches} found out of {total_possible_matches} possible)")
        print(f"Products with perfect recall: {sum(1 for r in recall_by_product.values() if r == 1.0)} out of {len(recall_by_product)}")
        print(f"Products with zero recall: {sum(1 for r in recall_by_product.values() if r == 0.0)} out of {len(recall_by_product)}")
        print(f"Using similarity threshold: {similarity_threshold}")
        
        # Print worst performing products
        if recall_by_product:
            worst_products = sorted(recall_by_product.items(), key=lambda x: x[1])[:3]
            print("\nWorst performing product clusters:")
            for product, recall in worst_products:
                print(f"  '{product}': {recall:.2f} recall")
        
        print("\n✅ Recall evaluation test completed")
        
    except Exception as e:
        print(f"❌ Error evaluating recall: {e}")
        raise

def run_all_tests():
    """Run all vector database tests"""
    print("Starting vector database tests...\n")
    
    try:
        # Test embedding model
        embedder = test_embedding_model()
        
        # Test enhanced descriptions
        test_enhanced_descriptions(embedder)
        
        # Test vector database creation
        vector_db = test_vector_db_creation()
        
        # Test similarity search
        test_similarity_search(vector_db)
        
        # Test performance metrics
        test_performance_metrics(vector_db)
        
        # Test recall using correct mapping
        test_mapping_recall(vector_db)
        
        print("\n=== All Tests Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
