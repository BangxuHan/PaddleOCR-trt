import logging


def get():
    return logging.getLogger('PaddleOCR')


def set_level(level):
    get().setLevel(level)
