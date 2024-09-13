from dataclasses import dataclass

@dataclass
class ConfigFullstack:
    tag: str
    dataset_name: str
    dataset_folder: str
    dataset_save: bool
    dataset_load_if_exists: bool
    model_name: str
    model_folder: str
    optimizer: str
    loss: str
    pretrained: str
    distance_metric: str
    epochs: int
    batch_size: int
    learning_rate: float
    ppi_reprezentation_size: int
    flowstats_reprezentation_size: int
    prototypes: int
    processed_reprezentation_size: int
    extra_layer: bool
    extra_layer_size: int
    prototype_dropout: float
    flowstats_dropout: float
    ppi_dropout: float
    prototype_hidden_size: int


