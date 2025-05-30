#!/usr/bin/env python3
"""
Interactive input utility for run_clustering.py script.
Provides functions for getting user input for various parameter types.
"""

import sys
import os

# Check if we're in an interactive environment
def is_interactive():
    """Check if we're running in an interactive environment where input is possible."""
    return sys.stdin.isatty() and os.isatty(0)

def get_yes_no_input(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user with default value."""
    default_str = "Y/n" if default else "y/N"
    
    if not is_interactive():
        print(f"{prompt} [{default_str}]: Using default: {default}")
        return default
    
    try:
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        
        if not response:
            return default
        
        return response[0] == 'y'
    except (EOFError, KeyboardInterrupt):
        print(f"\nInput interrupted. Using default: {default}")
        return default

def get_string_input(prompt: str, default: str, options: list = None) -> str:
    """Get string input from user with default value and optional list of choices."""
    if options:
        option_str = ", ".join(options)
        prompt = f"{prompt} (options: {option_str})"
    
    if not is_interactive():
        print(f"{prompt} [default: {default}]: Using default: {default}")
        return default
        
    try:
        response = input(f"{prompt} [default: {default}]: ").strip()
        
        if not response:
            return default
            
        if options and response not in options:
            print(f"Warning: '{response}' is not in the list of options: {option_str}")
            print(f"Using provided input anyway: '{response}'")
            
        return response
    except (EOFError, KeyboardInterrupt):
        print(f"\nInput interrupted. Using default: {default}")
        return default

def get_int_input(prompt: str, default: int, min_val: int = None, max_val: int = None) -> int:
    """Get integer input from user with default value and optional min/max values."""
    range_str = ""
    if min_val is not None and max_val is not None:
        range_str = f" (range: {min_val}-{max_val})"
    elif min_val is not None:
        range_str = f" (min: {min_val})"
    elif max_val is not None:
        range_str = f" (max: {max_val})"
        
    # In non-interactive mode, just return the default
    if not is_interactive():
        print(f"{prompt}{range_str} [default: {default}]: Using default: {default}")
        return default
        
    while True:
        try:
            response = input(f"{prompt}{range_str} [default: {default}]: ").strip()
            
            if not response:
                return default
                
            value = int(response)
            
            if min_val is not None and value < min_val:
                print(f"Error: Value must be at least {min_val}")
                continue
                
            if max_val is not None and value > max_val:
                print(f"Error: Value must be at most {max_val}")
                continue
                
            return value
        except ValueError:
            print("Error: Please enter a valid integer")
        except (EOFError, KeyboardInterrupt):
            print(f"\nInput interrupted. Using default: {default}")
            return default

def get_float_input(prompt: str, default: float, min_val: float = None, max_val: float = None) -> float:
    """Get float input from user with default value and optional min/max values."""
    range_str = ""
    if min_val is not None and max_val is not None:
        range_str = f" (range: {min_val}-{max_val})"
    elif min_val is not None:
        range_str = f" (min: {min_val})"
    elif max_val is not None:
        range_str = f" (max: {max_val})"
        
    # In non-interactive mode, just return the default
    if not is_interactive():
        print(f"{prompt}{range_str} [default: {default}]: Using default: {default}")
        return default
        
    while True:
        try:
            response = input(f"{prompt}{range_str} [default: {default}]: ").strip()
            
            if not response:
                return default
                
            value = float(response)
            
            if min_val is not None and value < min_val:
                print(f"Error: Value must be at least {min_val}")
                continue
                
            if max_val is not None and value > max_val:
                print(f"Error: Value must be at most {max_val}")
                continue
                
            return value
        except ValueError:
            print("Error: Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print(f"\nInput interrupted. Using default: {default}")
            return default
