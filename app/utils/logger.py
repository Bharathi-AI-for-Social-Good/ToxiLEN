import logging

def get_logger(name:str = "default", log_file = 'logs/run2.log', logging_level=logging.INFO ):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s") 
        fh = logging.StreamHandler()
        fh.setFormatter(formatter)
        
        sh = logging.FileHandler(log_file)
        sh.setFormatter(formatter)
        
        logger.addHandler(sh)
        logger.addHandler(fh)
        
        
    return logger
        