from keras import layers, models, Sequential, backend
import tensorflow as tf
import lib.ds_layer as ds_layer
import lib.utility_layer_train as utility_layer_train
import lib.utility_layer_test as utility_layer_test
import lib.AU_imprecision as AU_imprecision

@tf.keras.utils.register_keras_serializable()
class DstModel_FullStack_EarlyFusion(models.Model):
    def __init__(self,
                 prototypes: int, class_num: int,
                 ppi_reprezentation_size: int, flowstats_reprezentation_size: int,
                 processed_reprezentation_size: int,
                 distance_metric: str, extra_layer = False, extra_layer_size = 64,
                 prototype_dropout = 0.3, flowstats_dropout = 0.2, ppi_dropout = 0.2,
                 prototype_hidden_size = 64, *args, **kwargs):
        super(DstModel_FullStack_EarlyFusion, self).__init__(*args, **kwargs)
        self.prototypes = prototypes
        self.class_num = class_num
        self.ppi_reprezentation_size = ppi_reprezentation_size
        self.flowstats_reprezentation_size = flowstats_reprezentation_size
        self.processed_reprezentation_size = processed_reprezentation_size
        self.distance_metric = distance_metric
        self.extra_layer = extra_layer
        self.extra_layer_size = extra_layer_size
        self.prototype_dropout = prototype_dropout
        self.flowstats_dropout = flowstats_dropout
        self.ppi_dropout = ppi_dropout
        self.prototype_hidden_size = prototype_hidden_size
        self.ppi_modality = Sequential([
            layers.Input(shape=(4, 30), name="ppi_input"),
            layers.LayerNormalization(axis=1),
            layers.Conv1D(16, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(8, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(4, 3, activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(ppi_reprezentation_size, activation='relu'),
            layers.Dropout(ppi_dropout)
        ])
        self.flow_stats_modality = Sequential([
            layers.Input(shape=(66,)),
            layers.BatchNormalization(),
            layers.Dense(flowstats_reprezentation_size, activation='relu'),
            layers.Dropout(flowstats_dropout)
        ])
        self.extra = layers.Dense(extra_layer_size, activation='relu') if extra_layer else layers.Identity()
        self.prototype_layers = Sequential([
            layers.BatchNormalization(),
            self.extra,
            layers.Dense(prototype_hidden_size, activation='relu'),
            layers.Dropout(prototype_dropout),
            layers.Dense(self.processed_reprezentation_size, activation='relu'),
            layers.BatchNormalization(),
            ds_layer.DS1(self.prototypes, self.processed_reprezentation_size, distance_metric=self.distance_metric),
            ds_layer.DS1_activate(self.prototypes),
            ds_layer.DS2(self.prototypes, self.class_num),
            ds_layer.DS2_omega(self.prototypes, self.class_num)
        ])
        self.combinational_layers = Sequential([
            ds_layer.DS3_Dempster(self.prototypes, self.class_num),
            ds_layer.DS3_normalize()
        ])
        self.pignistic_transformation = utility_layer_train.DM_pignistic(class_num)
        self.decision_maker = None

    def call(self, x):
        ppi, flowstats = x
        x1 = self.ppi_modality(ppi)
        x2 = self.flow_stats_modality(flowstats)
        x = layers.Concatenate()([x1, x2])
        x = self.prototype_layers(x)
        x = self.combinational_layers(x)
        return self.pignistic_transformation(x)
    
    def get_combined_mass(self, x):
        ppi, flowstats = x
        x1 = self.ppi_modality(ppi)
        x2 = self.flow_stats_modality(flowstats)
        x = layers.Concatenate()([x1, x2])
        x = self.prototype_layers(x)
        return self.combinational_layers(x)
    
    def setup_dm(self, nu, utility_matrix):
        num_act_set = (2**self.class_num) - 1
        self.decision_maker = utility_layer_test.DM_test(self.class_num, num_act_set, nu)
        self.decision_maker.set_weights(tf.reshape(utility_matrix, [1, num_act_set, self.class_num]))

    def predict_dm(self, x):
        ppi, flowstats = x
        x1 = self.ppi_modality(ppi)
        x2 = self.flow_stats_modality(flowstats)
        x = layers.Concatenate()([x1, x2])
        x = self.prototype_layers(x)
        x = self.combinational_layers(x)
        return self.decision_maker(x)
    
    def get_config(self):
        config = super(DstModel_FullStack_EarlyFusion, self).get_config()
        config.update({
            'prototypes': self.prototypes,
            'class_num': self.class_num,
            'ppi_reprezentation_size': self.ppi_reprezentation_size,
            'flowstats_reprezentation_size': self.flowstats_reprezentation_size,
            'processed_reprezentation_size': self.processed_reprezentation_size,
            'distance_metric': self.distance_metric,
            'extra_layer': self.extra_layer,
            'extra_layer_size': self.extra_layer_size,
            'prototype_dropout': self.prototype_dropout,
            'flowstats_dropout': self.flowstats_dropout,
            'ppi_dropout': self.ppi_dropout,
            'prototype_hidden_size': self.prototype_hidden_size
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def f1(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=0)
    tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), 'float32'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float32'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float32'), axis=0)

    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())

    f1 = 2 * p * r / (p + r + backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return tf.reduce_mean(f1)

def f1_loss(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'), axis=0)
    tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), 'float32'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float32'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float32'), axis=0)

    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())

    f1 = 2 * p * r / (p + r + backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - tf.reduce_mean(f1)