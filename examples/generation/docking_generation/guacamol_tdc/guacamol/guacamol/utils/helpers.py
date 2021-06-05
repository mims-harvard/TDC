import logging


def setup_default_logger():
    """
    Call this function in your main function to initialize a basic logger.

    To have more control on the format or level, call `logging.basicConfig()` directly instead.

    If you don't initialize any logger, log entries from the guacamol package will not appear anywhere.
    """
    logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
