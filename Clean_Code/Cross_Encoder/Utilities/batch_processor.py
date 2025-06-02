"""
Batch processing utilities for Cross_Encoder module.
Handles efficient processing of data in batches with progress tracking.
"""

import os
import sys
from typing import List, Callable, Any, Dict, Optional
from tqdm import tqdm

def process_in_batches(items: List[Any], 
                      processor_func: Callable, 
                      batch_size: int = 32,
                      show_progress: bool = True,
                      progress_desc: str = "Processing batches") -> List[Any]:
    """
    Process a list of items in batches using the provided processor function.
    
    Args:
        items: List of items to process
        processor_func: Function that processes a batch of items and returns results
        batch_size: Size of batches to process
        show_progress: Whether to show a progress bar
        progress_desc: Description for the progress bar
        
    Returns:
        List of processed results in the same order as the input items
        
    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If batch processing fails
    """
    if not items:
        return []
    
    if not callable(processor_func):
        raise ValueError("processor_func must be a callable function")
    
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    
    # Calculate number of batches
    num_batches = (len(items) + batch_size - 1) // batch_size
    
    # Process in batches
    all_results = []
    
    # Setup progress bar if requested
    batch_range = tqdm(range(0, len(items), batch_size), total=num_batches, 
                       desc=progress_desc) if show_progress else range(0, len(items), batch_size)
    
    for batch_start in batch_range:
        batch_end = min(batch_start + batch_size, len(items))
        batch = items[batch_start:batch_end]
        
        try:
            # Process the batch
            batch_results = processor_func(batch)
            
            # Ensure we got the right number of results
            if len(batch_results) != len(batch):
                raise RuntimeError(
                    f"Batch processor returned {len(batch_results)} results for {len(batch)} items"
                )
            
            all_results.extend(batch_results)
        except Exception as e:
            # Provide more context for the error
            raise RuntimeError(f"Error processing batch {batch_start//batch_size + 1}/{num_batches}: {str(e)}")
    
    return all_results

# Test code
if __name__ == "__main__":
    print("Testing batch processor...")
    
    # Create test data
    test_data = list(range(100))
    
    # Define a simple processor function
    def square_items(items):
        return [x * x for x in items]
    
    # Process in batches
    batch_size = 10
    results = process_in_batches(test_data, square_items, batch_size=batch_size)
    
    print(f"Processed {len(test_data)} items in batches of {batch_size}")
    print(f"First 5 results: {results[:5]}")
    
    # Verify results
    expected = [x * x for x in test_data]
    assert results == expected, "Results don't match expected values"
    
    print("Batch processor tests passed!")
