import logging

def setup_logger(name):
    """Set up and return a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def print_section(title):
    """Print a clearly visible section title for output cleanliness."""
    print("\n" + "="*60)
    print(f"{title.center(60)}")
    print("="*60 + "\n")
