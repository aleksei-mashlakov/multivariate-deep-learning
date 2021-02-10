import logging
import logging.handlers

#
# This function is called in the main python file in order 
# to initialise logging.
# After that, in each other file, we just need to do the following:
#       import logging
#       log = logging.getLogger(name) where name is the same one used in LogInit
# 
# The arguments that this function take are:
# - name:         name of the logger. Passed from the main python file. The same must be used in all other files
# - filename:     full path of the file where logs are to be written
# - debuglevel:   Level for the FileHandler. Messages with level bigger or equal to this one will be logged
#
def LogInit(name, filename, debuglevel = logging.INFO, log = True):
    # If the passed debuglevel is not the allowed one, print an error and default to logging.INFO
    try:
        assert(debuglevel in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL])
    except AssertionError as err:
        logging.error("Invalid debuglevel (%d), changing it to %d", debuglevel, logging.INFO)
        debuglevel = logging.INFO

    # Create a logger with name 'name' and set its level to debuglevel`
    logger = logging.getLogger(name)
    logger.setLevel(debuglevel)

    # Set the log format
    formatter = logging.Formatter('%(asctime)s.%(msecs)d (%(process)d) (%(levelname)s) %(module)s.%(funcName)s -> %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Configure a StreamHandler that will log messages with level ERROR and CRITICAL onto the console.
    # Those messages will be logged even if log == False
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log == True:
        # Configure a TimedRotatingFileHandler that creates a new file at midnight in order to log
        # all messages with level greater than debuglevel if and only if log == True
        fh = logging.handlers.TimedRotatingFileHandler(filename, when='midnight')
        fh.setLevel(debuglevel)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        # NullHandler in case log == False
        nh = logging.NullHandler()
        logger.addHandler(nh)

    return logger
