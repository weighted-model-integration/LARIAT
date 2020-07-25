
import logging
from sys import stdout
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger("wmilearn")
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(stdout)
logger.addHandler(stream_handler)
