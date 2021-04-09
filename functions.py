"""Misc functions needed for the module"""
import logging

# {{{ def create_logger(logger_name, logger_level):
def create_logger(logger_name, logger_file, logger_level):
    """Creates a logger with a given name and specified logger level.

    Inputs:
    -------
    logger_name - str
        name of the logger
    logger_file - str
        file name of the logger
    logger_level -
        takes one of
        (logging.NOTSET,
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL)

    Outputs:
    --------
    logger

    """
    logger = logging.getLogger(logger_name)
    filehandler = logging.FileHandler(logger_file)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    logger.setLevel(logging.INFO)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger
# }}} create_logger(logger_name, logger_level)
