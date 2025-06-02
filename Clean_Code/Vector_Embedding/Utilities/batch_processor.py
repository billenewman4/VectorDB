"""
Batch processing utilities for handling large datasets.

This module provides functions for processing large datasets in batches,
which helps manage memory usage and provides progress tracking.
"""

import numpy as np
from typing import List, Any, Callable, TypeVar, Generic, Optional
from tqdm import tqdm

T = TypeVar('T')
R = TypeVar('R')


def process_in_batches(items: List[T], 
                      processor_func: Callable[[List[T]], List[R]], 
                      batch_size: int = 100,
                      show_progress: bool = True) -> List[R]:
    """
    Process a list of items in batches to manage memory and show progress.
    
    Args:
        items: List of items to process
        processor_func: Function that processes a batch of items and returns results
        batch_size: Number of items to process in each batch
        show_progress: Whether to show a progress bar
        
    Returns:
        List of processed results
        
    Raises:
        ValueError: If items is None or empty, or batch_size is invalid
        TypeError: If processor_func is not callable
        Exception: If batch processing fails
    """
    # Validate inputs
    if items is None:
        raise ValueError("Items list cannot be None")
    
    if not items:
        raise ValueError("Items list cannot be empty")
        
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")
        
    if not callable(processor_func):
        raise TypeError("Processor function must be callable")
    
    # Calculate number of batches
    num_items = len(items)
    num_batches = (num_items + batch_size - 1) // batch_size
    
    # Process in batches
    all_results = []
    
    # Set up progress tracking if requested
    iterator = range(0, num_items, batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=num_batches, desc="Processing batches")
    
    # Process each batch
    for i in iterator:
        end_idx = min(i + batch_size, num_items)
        batch = items[i:end_idx]
        
        try:
            # Process batch
            batch_results = processor_func(batch)
            
            # Validate batch results
            if batch_results is None:
                raise ValueError(f"Processor function returned None for batch {i//batch_size + 1}")
                
            # Add batch results to overall results
            all_results.extend(batch_results)
            
        except Exception as e:
            # Add batch information to the exception
            raise Exception(f"Error processing batch {i//batch_size + 1}/{num_batches} (items {i}-{end_idx-1}): {str(e)}") from e
    
    return all_results


if __name__ == "__main__":
    # Test the batch processor
    try:
        print("Testing batch processor...")
        
        # Define a simple processor function
        def square_numbers(batch: List[int]) -> List[int]:
            return [x * x for x in batch]
        
        # Create a test dataset
        test_data = list(range(1, 1001))
        
        # Process in batches
        results = process_in_batches(
            items=test_data,
            processor_func=square_numbers,
            batch_size=100,
            show_progress=True
        )
        
        # Verify results
        print(f"Processed {len(results)} items")
        print(f"First 5 results: {results[:5]}")
        print(f"Last 5 results: {results[-5:]}")
        
        # Verify correctness
        expected = [x * x for x in test_data]
        assert results == expected, "Results don't match expected values"
        
        print("Batch processor tests passed!")
    except Exception as e:
        print(f"Error testing batch processor: {e}")
