import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class DM_pignistic(tf.keras.layers.Layer):
    def __init__(self, num_class, **kwargs):
        super(DM_pignistic, self).__init__(**kwargs)
        self.num_class = num_class

    def call(self, inputs):
        aveage_Pignistic = inputs[:, -1] / self.num_class
        aveage_Pignistic = tf.expand_dims(aveage_Pignistic, -1)
        Pignistic_prob = inputs[:, :] + aveage_Pignistic
        Pignistic_prob = Pignistic_prob[:, 0:-1]
        return Pignistic_prob
