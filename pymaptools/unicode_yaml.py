"""
Overrides the default string handling function in PyYAML
to always return unicode objects

See http://stackoverflow.com/a/2967461/597371
"""
import yaml
from yaml import Loader, SafeLoader


def construct_yaml_str(self, node):
    return self.construct_scalar(node)


Loader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_str)
SafeLoader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_str)
