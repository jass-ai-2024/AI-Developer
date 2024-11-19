import logging
import sys

LOGGER = logging.getLogger("logger")
LOGGER.setLevel(logging.INFO)

console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter(fmt="[%(asctime)s: %(levelname)s] %(message)s")
)
LOGGER.addHandler(console_handler)

# file_handler = logging.FileHandler('./logs/agent_endpoint.log')
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(logging.Formatter(fmt="[%(asctime)s: %(levelname)s] %(message)s"))
# LOGGER.addHandler(file_handler)
