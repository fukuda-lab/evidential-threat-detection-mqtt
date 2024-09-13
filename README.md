# Evidential classifier: MQTT Case


## Datasets
For creating datasets, we used [`ipfixprobe`](https://github.com/CESNET/ipfixprobe) exporter with conjunction with  [Nemea framework](https://github.com/cesnet/nemea), using following commands:
```bash
ipfixprobe -i "pcap;file=<filename.pcap>" -p "pstats" -p "phists" -p "mqtt" -o "unirec;i=f:<filename.pcap>.trapcap:timeout=WAIT;p=(pstats,phists,mqtt)" # export pcaps to UniRec format

for i in *.trapcap; do /usr/bin/nemea/logger -t -i f:$i -w $i.csv; done # batch export UniRec to CSV
```

Then CSV should be saved to `datasets` folder (or any other folder specified in experiment configuration). Since the files are too large to be uploaded to the repository, they are not included in the repository.

## Configuration description

In folder `conf`, there are two types of configuration files:
1. *-eval.yaml - configuration for single evaluation of the model, during the hyperparameters search, the values are overwrite by the values from the hyperparameters search configuration file by the control server. The configuration is used with [`hydra`](https://hydra.cc/docs/intro/) configuration system. For convenient usage, we use python dataclass to define the configuration schema in file `config.py`.
2. *-sweep.yaml - configuration for hyperparameters search, the values are used by the control server to create the hyperparameters search space, using [`wandb`](https://github.com/wandb/wandb) sweeps.

### Eval files description
```python
class ConfigFullstack:
    tag: str # tag for the experiment
    dataset_name: str # name of the dataset
    dataset_folder: str # folder with the dataset (relative to the project root)
    dataset_save: bool # save the dataset to the .pkl file for faster loading
    dataset_load_if_exists: bool # load the dataset from the .pkl file if exists           
    model_name: str # name of the model
    model_folder: str # folder with the model (relative to the project root)
    optimizer: str # optimizer used for training (adam, etc.)
    loss: str # loss function used for training (crossentropy, focal, f1_loss)
    pretrained: str # path to the pretrained model
    distance_metric: str # distance metric used for the prototype layer (euclidean, cosine, manhattan)
    epochs: int # number of epochs
    batch_size: int # batch size
    learning_rate: float # learning rate
    ppi_reprezentation_size: int # size of the PPI reprezentation
    flowstats_reprezentation_size: int # size of the flowstats reprezentation
    prototypes: int # number of prototypes
    processed_reprezentation_size: int # size of the processed reprezentation
    extra_layer: bool # use extra layer for reprezentation processing
    extra_layer_size: int # size of the extra layer, used if extra_layer is True
    prototype_dropout: float # dropout for the prototype layer
    flowstats_dropout: float # dropout for the flowstats layer
    ppi_dropout: float # dropout for the PPI layer
    prototype_hidden_size: int # size of the prototype hidden layer
```

### Sweep files description
```yaml
program: paper_evaluation.py # name of the pytohn program to run
method: bayes # method used for the hyperparameters search
metric:
  name: metric # metric used for the hyperparameters search, we named it "metric" (it is macro f1)
  goal: maximize # goal of the metric
parameters:
  loss:
    values: ["crossentropy", "f1", "focal"] # values for the loss function
  distance_metric:
    values: ["manhattan", "cosine", "euclidean"] # values for the distance metric
  batch_size:
    min: 4096
    max: 131072
  learning_rate:
    min: 0.00001
    max: 0.001
  ppi_reprezentation_size:
    min: 1
    max: 64
  flowstats_reprezentation_size:
    min: 5
    max: 64
  processed_reprezentation_size:
    min: 4
    max: 48
  prototypes:
    min: 15
    max: 75
  extra_layer:
    values: [true, false]
  extra_layer_size:
    min: 8
    max: 128
  prototype_dropout:
    min: 0.0
    max: 0.9
  flowstats_dropout:
    min: 0.0
    max: 0.9
  ppi_dropout:
    min: 0.0
    max: 0.9
  prototype_hidden_size:
    min: 8
    max: 128

command: # command to run the program
  - python
  - ${program}
  - "--config-name"
  - "paper-edge-eval"
```

## Running the experiments
First, we need to install the requirements:
```bash
pip install -r requirements.txt
```

First, we start and define wandb controller, for hyperparameters search:
```bash
wandb sweep --project paper-project conf/paper-edge-sweeper.yaml
```

Then, we might run multiple agents to run the experiments; you obtain `<ID>` from the previous command:
```bash
wandb agent paper-project/<ID> 
```

If you want directly run the experiment without hyperparameters search, you can run the following command, which run the experiment with the configuration file:
```bash
python paper_evaluation.py --config conf/paper-edge-eval.yaml
```

## Model

```
+----------------+      +-------------------+
| PPI (4x30)     |      | Flow Stats (66)   |
+----------------+      +-------------------+
       |                        |
       V                        V
+------------------+    +-----------------------+
| PPI Modality     |    | Flow Stats Modality   |
|                  |    |                       |
| LayerNorm        |    | BatchNorm             |
| Conv1D → MaxPool |    | Dense                 |
| Conv1D → MaxPool |    | Dropout               |
| Flatten          |    +-----------------------+
| Dense → Dropout  |            |   
+------------------+            |
       |                        |
       +----------+  +----------+
                  V  V
           +--------------------+
           |   Concatenation     |
           +--------------------+
                   |
                   V
       +---------------------------+
       |   Prototype Layers         |
       |   BatchNorm                |
       |   Extra Layer (optional)   |
       |   Dense → Dropout          |
       |   DS1 → DS1_activate       |
       |   DS2 → DS2_omega          |
       +---------------------------+
                   |
                   V
       +---------------------------+
       |   Combination Layers       |
       |   DS3_Dempster             |
       |   DS3_normalize            |
       +---------------------------+
                   |
                   V
       +---------------------------+
       |  Pignistic Transformation |
       |   (Mass to Probability)   |
       |      Use for training     | 
       +---------------------------+
                   |
                   V
               Final Output
```