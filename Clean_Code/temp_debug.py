#!/usr/bin/env python
import sys
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Import and run the main function from hierarchical clustering
        from Pipeline.main_hierarchical_clustering import main
        logger.debug("Starting main function")
        main()
    except Exception as e:
        logger.error(f"ERROR TYPE: {type(e).__name__}")
        logger.error(f"ERROR MESSAGE: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

if __name__ == "__main__":
    main()
