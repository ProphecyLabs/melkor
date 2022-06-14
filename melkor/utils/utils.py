import yaml
from collections import defaultdict


def to_snake(str_in, scream=False):
    """Convert string to snake_case or SNAKE_CASE"""
    # TODO: extend functionality to other characters that might need to be eliminated or replaced

    str_in = str_in.strip().replace(" ", "_").replace(",", "_").replace(":", "")

    if scream == False:
        return str_in.lower()
    else:
        return str_in.upper()


def config_parser(path: str):

    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return defaultdict(dict, data)
