from pathlib import Path
from typing import Union
from box import Box
import yaml

class ConfigParser:
    """
    Loading and parsing YAML files to configuration
    """

    @staticmethod
    def load(path: Union[str, Path]) -> Box:
        """
        Load a YAML configuration file from config directory given by a Union and return the list to Box object.
        This Box object will be frozen and will not allow change auto.
        :param path: The path of the YAML config file
        :return: Box object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file {path} does not exist.")

        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return Box(config_dict, frozen_box=True, default_box=False)