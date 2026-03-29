# Lightning modules
from .multimodal_lightning import LightningModel as MultimodalLightningModel
from .dna_only_lightning import LightningModel as DNAOnlyLightningModel

__all__ = [
    'MultimodalLightningModel',
    'DNAOnlyLightningModel',
]
