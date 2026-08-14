from .common import BaseTableEncoder, TableEncoding
from .bert_baseline import BertTableEncoder
from .tabbie import TabbieTableEncoder
from .strubert import StruBertTableEncoder
from .tapas_encoder import TapasTableEncoder
from .turl import TurlTableEncoder
from .hytrel import HyTrelTableEncoder

__all__ = [
    "BaseTableEncoder",
    "TableEncoding",
    "BertTableEncoder",
    "TabbieTableEncoder",
    "StruBertTableEncoder",
    "TapasTableEncoder",
    "TurlTableEncoder",
    "HyTrelTableEncoder",
]

ENCODER_REGISTRY = {
    "bert": BertTableEncoder,
    "tabbie": TabbieTableEncoder,
    "strubert": StruBertTableEncoder,
    "tapas": TapasTableEncoder,
    "turl": TurlTableEncoder,
    "hytrel": HyTrelTableEncoder,
}
