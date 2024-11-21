import logging

# Настройка логгера
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Создание обработчика для вывода в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Создание форматтера
formatter = logging.Formatter('[%(asctime)s: %(levelname)s] %(message)s')
console_handler.setFormatter(formatter)

# Добавление обработчика к логгеру
LOGGER.addHandler(console_handler)
