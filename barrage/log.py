import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
logger.propagate = False

console = logging.StreamHandler()
formatter = logging.Formatter(
    "Barrage %(asctime)s %(levelname)-5.4s: %(message)s", datefmt="%m/%d/%y %I:%M:%S"
)
console.setFormatter(formatter)
logger.addHandler(console)
