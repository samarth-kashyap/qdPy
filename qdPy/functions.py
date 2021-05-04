"""Misc functions needed for the module"""
import logging
import argparse

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
    logger.setLevel(logger_level)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger
# }}} create_logger(logger_name, logger_level)

# {{{ def create_logger(logger_name, logger_level):
def create_logger_stream(logger_name, logger_file, logger_level):
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
    logger.setLevel(logger_level)
    sh = logging.StreamHandler()
    sh.setLevel(logger_level)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
# }}} create_logger(logger_name, logger_level)


# {{{ def create_argparser():
def create_argparser():
    """Creates argument parser for arguments passed during
    execution of script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n0", help="radial order", type=int)
    parser.add_argument("--l0", help="angular degree", type=int)
    parser.add_argument("--lmin", help="min angular degree", type=int)
    parser.add_argument("--lmax", help="max angular degree", type=int)
    parser.add_argument("--maxiter", help="max MCMC iterations",
                        type=int)
    parser.add_argument("--precompute",
                        help="precompute the integrals upto r=0.9",
                        action="store_true")
    parser.add_argument("--use_precomputed",
                        help="use precomputed integrals",
                        action="store_true")
    parser.add_argument("--usempi",
                        help='use MPI for Bayesian estimation',
                        action="store_true")
    parser.add_argument("--parallel",
                        help='parallel processing',
                        action="store_true")
    args = parser.parse_args()
    return args
# }}} create_argparser()

# functions to create the extreme profiles near the surface to get the 
# range of spline coefficients for MCMC simulations
def get_matching_function(r, r_th):
    return 0.5*(np.tanh((r-r_th)/0.05) + 1)

def create_nearsurface_profile(r, r_th, w_dpt, which_ex='upex'):
    w_new = np.zeros_like(w_dpt)
    matching_function = get_matching_function(r, r_th)
    
    # nea surface enhanced or suppressed profile
    if(which_ex == 'upex'): w_new = matching_function * (fac_up * w_dpt)
    else: w_new = matching_function * (fac_lo * w_dpt)
        
    # adding the complementary part below the r_th
    w_new += (1 - matching_function) * w_dpt
    
    return w_new

