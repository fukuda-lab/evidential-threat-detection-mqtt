import pandas as pd
import numpy as np
import tensorflow as tf
from keras import models, metrics, callbacks, optimizers, utils, losses
import architectures
import hydra
from omegaconf import DictConfig, OmegaConf
import lib.utility_layer_test as utility_layer_test
from sklearn.model_selection import train_test_split
import logging
from sklearn.preprocessing import LabelEncoder
import wandb
from sklearn.utils import class_weight
import os

from config import ConfigFullstack
from constants import LABEL, FLOW_STATS_COLUMNS
import utilities
from data_processing import get_ppi_matrices, get_dataset

logging.basicConfig(level=logging.INFO)

LABEL = "LABEL"

# for reproducibility
utils.set_random_seed(0xdeadbeef)

@hydra.main(version_base=None, config_path="conf")
def main(cfg : DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = ConfigFullstack(**cfg_dict)

    os.environ["WANDB_INIT_TIMEOUT"] = "300"
    os.environ["WANDB_HTTP_TIMEOUT"] = "300"
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    run = wandb.init(
        project=f"{config.tag}",
        dir=os.path.expanduser("~"),
        config=utilities.flatten_dict(cfg_dict)
    )

    config = dict(cfg_dict)
    config.update(wandb.config)
    config = ConfigFullstack(**config)

    logging.info(f"Config: {config}")

    timestamp = run.start_time
    experiment_nickname = f"{run.name}.{timestamp}"

    dataset = get_dataset(config)

    class_count = dataset["LABEL"].nunique()
    labels = dataset[LABEL].values
    # encode labels to int
    labels_encoder = LabelEncoder()
    labels_encoded = labels_encoder.fit_transform(labels)
    label_names = labels_encoder.classes_

    wandb.config.update({
        "class_count": class_count,
        "experiment_nickname": experiment_nickname,
        "label_names": label_names
    })
    wandb.log({"misc/config": wandb.Html("<pre>" + "\n".join(f"{k}: {v}" for k,v in vars(config).items()) + "</pre>")}, step=0)

    logging.info(f"Class count: {class_count}")
    logging.info(f"Label names: {label_names}")


    # Split dataset into train, validation and test sets
    train_indices, test_indices = train_test_split(dataset.index, test_size=0.3, stratify=labels_encoded)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.15, stratify=labels_encoded[train_indices])


    value_count_train = pd.Series(labels_encoded[train_indices]).value_counts()
    value_count_val = pd.Series(labels_encoded[val_indices]).value_counts()
    value_count_test = pd.Series(labels_encoded[test_indices]).value_counts()
    value_count_full = pd.Series(labels_encoded).value_counts()

    logging.info(f"Full set value counts: {value_count_full}")
    logging.info(f"Train set value counts: {value_count_train}")
    logging.info(f"Validation set value counts: {value_count_val}")
    logging.info(f"Test set value counts: {value_count_test}")

    ####################
    # Prepare PPI data #
    ####################
    train_ppi, val_ppi, test_ppi = get_ppi_matrices(dataset, train_indices, val_indices, test_indices)

    ##########################
    # Prepare FlowStats data #
    ##########################
    train_flow_stats = dataset.loc[train_indices, FLOW_STATS_COLUMNS]
    val_flow_stats = dataset.loc[val_indices, FLOW_STATS_COLUMNS]
    test_flow_stats = dataset.loc[test_indices, FLOW_STATS_COLUMNS]

    # deleting dataset to free up memory
    del dataset

    # class weights
    logging.info("Calculating class weights")
    class_weights_calculate = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = {i: class_weights_calculate[i] for i in range(len(class_weights_calculate))}
    logging.info(f"Class weights: {class_weights}")
    
    ################
    # DST training #
    ################
    if len(config.pretrained) > 0:
        dst_model = models.load_model(f"{config.model_folder}/{config.pretrained}")
    else:
        logging.info("DST model train: Training DST model")

        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001
        )

        dst_model = architectures.DstModel_FullStack_EarlyFusion(
            prototypes=config.prototypes,
            class_num=class_count,
            distance_metric=config.distance_metric,
            ppi_reprezentation_size=config.ppi_reprezentation_size,
            flowstats_reprezentation_size=config.flowstats_reprezentation_size,
            processed_reprezentation_size=config.processed_reprezentation_size,
            extra_layer=config.extra_layer,
            extra_layer_size=config.extra_layer_size,
            prototype_dropout=config.prototype_dropout,
            flowstats_dropout=config.flowstats_dropout,
            ppi_dropout=config.ppi_dropout,
            prototype_hidden_size=config.prototype_hidden_size
        )
        if config.optimizer == 'adam':
            logging.info("DST model train: Using Adam optimizer")
            optimizer = optimizers.Adam(learning_rate=config.learning_rate)
        elif config.optimizer == 'sgd':
            logging.info("DST model train: Using SGD optimizer")
            optimizer = optimizers.SGD(learning_rate=config.learning_rate)
        else: # Default
            logging.info("DST model train: Using Adam optimizer (DEFAULTED)")
            optimizer = optimizers.Adam(learning_rate=config.learning_rate)
        
        if config.loss == 'crossentropy':
            logging.info("DST model train: Using Categorical Crossentropy loss")
            loss = losses.CategoricalCrossentropy()
        elif config.loss == 'focal':
            logging.info("DST model train: Using Focal Crossentropy loss")
            loss = losses.CategoricalFocalCrossentropy()
        elif config.loss == 'f1':
            logging.info("DST model train: Using F1 loss")
            loss = architectures.f1_loss
        else: # Default
            logging.info("DST model train: Using Categorical Crossentropy loss (DEFAULTED)")
            loss = losses.CategoricalCrossentropy()
        
        logging.info("DST model train: Compiling model")
        dst_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', metrics.F1Score(average="macro")]              
        )
        logging.info("DST model train: Fitting model")
        dst_model.fit(
            (train_ppi, train_flow_stats),
            utils.to_categorical(labels_encoded[train_indices]),
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=((val_ppi, val_flow_stats), utils.to_categorical(labels_encoded[val_indices])),
            shuffle=True,
            verbose=1,
            callbacks=[utilities.WandbCustomCallback(model_name="dst_model"), early_stopping],
            class_weight=class_weights
        )
        try:
            logging.info("DST model train: Saving model")
            dst_model.save(f"{config.model_folder}/{experiment_nickname}.dst_model.keras")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    # Testing pignistic transformation 
    logging.info("DST model test: Testing pignistic transformation")
    dst_model_pignistic_pred = dst_model.predict((test_ppi, test_flow_stats), batch_size=config.batch_size)
    utilities.wandb_classification_report(labels_encoded[test_indices], dst_model_pignistic_pred.argmax(axis=1), label_names, "dst_model_pignistic")

    # Testing DST model
    logging.info("DST model test: Testing DST model set-value inference")
    combined_masses = dst_model.get_combined_mass((test_ppi, test_flow_stats))
    result_table = []
    for tol_i in [0,1,2,3,4]:
        utility_matrix, act_set = utilities.generate_utility_matrix(num_class=class_count, tol_i=tol_i)
        for nu in [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]:
            dm = utility_layer_test.DM_test(num_class=class_count, num_set=len(act_set), nu=nu)
            dm.set_weights(tf.reshape(utility_matrix, [1, len(act_set), class_count]))
            decisions = dm(combined_masses)
            dm_test_output_max = tf.argmax(decisions, axis=-1).numpy()
            imprecise_results = [act_set[act] for act in dm_test_output_max]
            result = utilities.set_value_evaluation(labels_encoded[test_indices], imprecise_results, dm_test_output_max, utility_matrix, act_set, labels_encoder.classes_)
            result["tolerance"] = tol_i
            result["nu"] = nu
            result_table.append(result)
    result_table = pd.DataFrame(result_table)
    wandb.log({"Evaluation": wandb.Table(dataframe=result_table)})
    run.finish()

if __name__ == "__main__":
    main()

