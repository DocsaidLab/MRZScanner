from importlib.metadata import version

from .scanner import ErrorCodes, ModelType, MRZScanner, SpottingInference
from .utils import replace_digits, replace_letters, replace_sex

__version__ = version("mrzscanner_docsaid")
