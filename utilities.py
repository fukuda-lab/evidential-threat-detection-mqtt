import numpy as np
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from keras import models, utils, callbacks
import wandb
import pandas as pd
import tempfile
import os
import tensorflow as tf
from wandb.sdk.lib import telemetry
from sklearn.metrics import classification_report, confusion_matrix, precision_score, f1_score, recall_score, accuracy_score
from set_val_inference import PowerSetsBinary, func, cons1, cons2
from scipy.optimize import minimize
from evidential_layers.AU_imprecision import average_utility

def flatten_dict(d, parent_key='', sep='.'):
    return {f'{parent_key}{sep}{k}' if parent_key else k: v
            for kk, vv in d.items()
            for k, v in (flatten_dict(vv, f'{parent_key}{sep}{kk}' if parent_key else kk, sep=sep) if isinstance(vv, dict) else {kk: vv}).items()}


class WandbCustomCallback(callbacks.Callback):
    """Logger that sends system metrics to W&B.

    `WandbMetricsLogger` automatically logs the `logs` dictionary that callback methods
    take as argument to wandb.

    This callback automatically logs the following to a W&B run page:
    * system (CPU/GPU/TPU) metrics,
    * train and validation metrics defined in `model.compile`,
    * learning rate (both for a fixed value or a learning rate scheduler)

    Notes:
    If you resume training by passing `initial_epoch` to `model.fit` and you are using a
    learning rate scheduler, make sure to pass `initial_global_step` to
    `WandbMetricsLogger`. The `initial_global_step` is `step_size * initial_step`, where
    `step_size` is number of training steps per epoch. `step_size` can be calculated as
    the product of the cardinality of the training dataset and the batch size.

    Arguments:
        log_freq: ("epoch", "batch", or int) if "epoch", logs metrics
            at the end of each epoch. If "batch", logs metrics at the end
            of each batch. If an integer, logs metrics at the end of that
            many batches. Defaults to "epoch".
        initial_global_step: (int) Use this argument to correctly log the
            learning rate when you resume training from some `initial_epoch`,
            and a learning rate scheduler is used. This can be computed as
            `step_size * initial_step`. Defaults to 0.
    """

    def __init__(
        self,
        model_name,
        log_freq = "epoch",
        initial_global_step: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` before WandbMetricsLogger()"
            )

        with telemetry.context(run=wandb.run) as tel:
            tel.feature.keras_metrics_logger = True

        if log_freq == "batch":
            log_freq = 1

        self.model_name = model_name
        self.logging_batch_wise = isinstance(log_freq, int)
        self.log_freq = log_freq if self.logging_batch_wise else None
        self.global_batch = 0
        self.global_step = initial_global_step

        if self.logging_batch_wise:
            # define custom x-axis for batch logging.
            wandb.define_metric("batch/batch_step")
            # set all batch metrics to be logged against batch_step.
            wandb.define_metric("batch/*", step_metric="batch/batch_step")
        else:
            # define custom x-axis for epoch-wise logging.
            wandb.define_metric(f"{self.model_name}/epoch")
            # set all epoch-wise metrics to be logged against epoch.
            wandb.define_metric(f"{self.model_name}/*", step_metric=f"{self.model_name}/epoch")

    def _get_lr(self):
        if isinstance(self.model.optimizer.learning_rate, tf.Variable):
            return float(self.model.optimizer.learning_rate.numpy().item())
        try:
            return float(
                self.model.optimizer.learning_rate(step=self.global_step).numpy().item()
            )
        except Exception:
            wandb.termerror("Unable to log learning rate.", repeat=False)
            return None

    def on_epoch_end(self, epoch: int, logs = None) -> None:
        """Called at the end of an epoch."""
        logs = dict() if logs is None else {f"{self.model_name}/{k}": v for k, v in logs.items()}

        logs[f"{self.model_name}/epoch"] = epoch

        lr = self._get_lr()
        if lr is not None:
            logs[f"{self.model_name}/learning_rate"] = lr

        wandb.log(logs)

    def on_batch_end(self, batch: int, logs = None) -> None:
        self.global_step += 1
        """An alias for `on_train_batch_end` for backwards compatibility."""
        if self.logging_batch_wise and batch % self.log_freq == 0:
            logs = {f"batch/{k}": v for k, v in logs.items()} if logs else {}
            logs["batch/batch_step"] = self.global_batch

            lr = self._get_lr()
            if lr is not None:
                logs["batch/learning_rate"] = lr

            wandb.log(logs)

            self.global_batch += self.log_freq

    def on_train_batch_end(
        self, batch: int, logs = None
    ) -> None:
        """Called at the end of a training batch in `fit` methods."""
        self.on_batch_end(batch, logs if logs else {})


import numpy as np
from sklearn.metrics import classification_report, precision_score
import wandb

def wandb_classification_report(true_labels, pred_labels, target_names, name, only_binary=False):
    report_columns = ["Class", "Precision", "Recall", "F1-score", "Support"]

    benign_label = np.where(target_names == 'benign')[0][0]
    binary_real = [0 if label == benign_label else 1 for label in true_labels]
    binary_pred = [0 if label == benign_label else 1 for label in pred_labels]
    malicious_precision = precision_score(binary_real, binary_pred, zero_division=0)

    if only_binary:
        wandb.log({
            f"metrics-{name}/malicious-precision": malicious_precision,
            f"metrics-{name}/malicious-cm": wandb.plot.confusion_matrix(y_true=binary_real, preds=binary_pred, class_names=['benign', 'malicious'])
        })
        return

    # Get the unique classes present in true_labels and pred_labels
    present_classes = np.unique(np.concatenate([true_labels, pred_labels]))

    # Filter the target_names and true_labels, pred_labels accordingly
    filtered_target_names = [target_names[i] for i in present_classes]
    filtered_true_labels = [present_classes.tolist().index(label) for label in true_labels]
    filtered_pred_labels = [present_classes.tolist().index(label) for label in pred_labels]

    class_report = classification_report(filtered_true_labels, filtered_pred_labels, target_names=filtered_target_names, zero_division=0).splitlines()

    report_table = []
    for line in class_report[2:(len(filtered_target_names) + 2)]:
        report_table.append(line.split())

    macro_f1_score = f1_score(filtered_true_labels, filtered_pred_labels, average='macro')

    wandb.log({
        f"metrics-{name}/cm": wandb.plot.confusion_matrix(y_true=filtered_true_labels, preds=filtered_pred_labels, class_names=filtered_target_names),
        f"metrics-{name}/report": wandb.Table(data=report_table, columns=report_columns),
        f"metrics-{name}/macro_f1": macro_f1_score,
        f"metrics-{name}/malicious-precision": malicious_precision,
        f"metrics-{name}/malicious-cm": wandb.plot.confusion_matrix(y_true=binary_real, preds=binary_pred, class_names=['benign', 'malicious'])
    })

    wandb.log({
        "metric": macro_f1_score
    })

def calculate_map(imprecise_results, labels_encoded):
    true_labels = np.array(labels_encoded)
    average_precisions = []

    for i, prediction_set in enumerate(imprecise_results):
        # Convert prediction set to a set of integers
        true_label = true_labels[i]
        # Calculate precision
        if true_label in prediction_set:
            precision = 1.0 / len(prediction_set)
        else:
            precision = 0.0
        average_precisions.append(precision)
    # Calculate mean average precision
    mean_ap = np.mean(average_precisions)
    return mean_ap

def generate_utility_matrix(num_class: int, tol_i: int) -> np.ndarray:
    """
    tol_i = 0 with tol=0.5, tol_i = 1 with tol=0.6, tol_i = 2 with tol=0.7, tol_i = 3 with tol=0.8, tol_i = 4 with tol=0.9
    """
    for j in range(2,(num_class+1)):
        num_weights = j
        ini_weights = np.asarray(np.random.rand(num_weights))

        name='weight'+str(j)
        locals()['weight'+str(j)]= np.zeros([5, j])

        for i in range(5):
            tol = 0.5 + i * 0.1

            cons = ({'type': 'eq', 'fun' : lambda x: cons1(x)-1},
                {'type': 'eq', 'fun' : lambda x: cons2(x)-tol},
                {'type': 'ineq', 'fun' : lambda x: x-0.00000001}
                )
        
            res = minimize(func, ini_weights, method='SLSQP', options={'disp': True}, constraints=cons)
            locals()['weight'+str(j)][i] = res.x

    class_set=list(range(num_class))
    act_set= PowerSetsBinary(class_set)
    act_set.remove(act_set[0]) # empty set is not needed
    act_set=sorted(act_set)

    utility_matrix = np.zeros([len(act_set), len(class_set)])
    for i in range(len(act_set)):
        intersec = class_set and act_set[i]
        if len(intersec) == 1:
            utility_matrix[i, intersec] = 1
        
        else:
            for j in range(len(intersec)):
                utility_matrix[i, intersec[j]] = locals()['weight'+str(len(intersec))][tol_i, 0]

    return utility_matrix, act_set

def set_value_evaluation(y_true, y_pred, y_pred_dm_output, utility_matrix, act_set, target_names):
    """
    y_true: true labels
    y_pred: predicted labels
    utility_matrix: utility matrix
    """
    au = average_utility(utility_matrix, y_pred_dm_output, y_true, act_set)
    mean_ap = calculate_map(y_pred, y_true)

    single_prec_pred = np.array([res[0] if len(res) == 1 else -1 for res in y_pred])
    valid_mask = single_prec_pred != -1
    rejected = len(y_pred) - np.sum(valid_mask)
    masked_data = [item for item, valid in zip(y_pred, valid_mask) if valid]
    masked_data = np.array(masked_data).flatten()

    report_columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
    benign_label = np.where(target_names == 'benign')[0][0]
    binary_real = [0 if label == benign_label else 1 for label in y_true[valid_mask]]
    binary_pred = [0 if label == benign_label else 1 for label in masked_data]
    malicious_precision = precision_score(binary_real, binary_pred, zero_division=0)

    macro_f1  = f1_score(y_true[valid_mask], masked_data, average='macro')
    macro_precision = precision_score(y_true[valid_mask], masked_data, average='macro')
    macro_recall = recall_score(y_true[valid_mask], masked_data, average='macro')
    accuracy_score_ = accuracy_score(y_true[valid_mask], masked_data)
            
    return {
        "average_utility": au,
        "map": mean_ap,
        "malprec": malicious_precision,
        "single_f1": macro_f1,
        "single_precision": macro_precision,
        "single_recall": macro_recall,
        "accuracy_score": accuracy_score_,
        "rejected": rejected,
        "rejected_rate": rejected / len(y_pred)
    }
