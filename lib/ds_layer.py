import tensorflow as tf
from keras import initializers

@tf.keras.utils.register_keras_serializable()
class DS1(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, distance_metric='euclidean', **kwargs):
        super(DS1, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.distance_metric = distance_metric.lower()
        self.w = self.add_weight(
            name='Prototypes',
            shape=(units, input_dim),
            initializer=initializers.HeNormal(),
            trainable=True
        )        

    def call(self, inputs):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(inputs)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(inputs)
        elif self.distance_metric == 'cosine':
            return self.cosine_distance(inputs)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def euclidean_distance(self, inputs):
        un_mass = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.w), axis=-1)
        return un_mass

    def manhattan_distance(self, inputs):
        un_mass = tf.reduce_sum(tf.abs(tf.expand_dims(inputs, axis=1) - self.w), axis=-1)
        return un_mass

    def cosine_distance(self, inputs):
        normalized_inputs = tf.nn.l2_normalize(inputs, axis=-1)
        normalized_w = tf.nn.l2_normalize(self.w, axis=-1)
        cosine_similarity = tf.reduce_sum(normalized_inputs[:, tf.newaxis] * normalized_w, axis=-1)
        un_mass = 1 - cosine_similarity
        return un_mass

    def get_config(self):
        config = super(DS1, self).get_config()
        config.update({
            'units': self.units,
            'input_dim': self.input_dim,
            'distance_metric': self.distance_metric
        })
        return config
    
@tf.keras.utils.register_keras_serializable()
class DS1_activate(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super(DS1_activate, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.xi = self.add_weight(
            name='xi',
            shape=(1, input_dim),
            initializer=initializers.HeNormal(),
            trainable=True
        )
        self.eta = self.add_weight(
            name='eta',
            shape=(1, input_dim),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        # Vectorized computation for gamma and alpha
        gamma = tf.square(self.eta)
        alpha = 1 / (tf.exp(-self.xi) + 1)
        
        # Compute si with vectorized operations
        si = gamma * inputs
        si = tf.exp(-si) * alpha
        
        # Normalize si along the last dimension
        si_max = tf.reduce_max(si, axis=-1, keepdims=True)
        si = si / (si_max + 0.0001)
        
        return si

    def get_config(self):
        config = super(DS1_activate, self).get_config()
        config.update({
            'input_dim': self.input_dim
        })
        return config
    

class DS2(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class, **kwargs):
        super(DS2, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.num_class = num_class
        self.beta = self.add_weight(
            name='beta',
            shape=(input_dim, num_class),
            initializer=initializers.HeNormal(),
            trainable=True
        )

    def call(self, inputs):
        beta = tf.square(self.beta)
        beta_sum = tf.reduce_sum(beta, -1, keepdims=True)
        u = beta / beta_sum

        # Ensure inputs are expanded correctly
        inputs_expanded = tf.expand_dims(inputs, -1)
        mass_prototype = u * inputs_expanded
        return mass_prototype

    def get_config(self):
        config = super(DS2, self).get_config()
        config.update({'input_dim': self.input_dim, 'num_class': self.num_class})
        return config
    
@tf.keras.utils.register_keras_serializable()
class DS2_omega(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class, **kwargs):
        super(DS2_omega, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.num_class = num_class

    def call(self, inputs):
        # Assuming inputs have shape (batch_size, proto_num, num_class)
        mass_omega_sum = tf.reduce_sum(inputs, axis=-1, keepdims=True)  # Shape: (batch_size, proto_num, 1)
        mass_omega_sum = 1.0 - mass_omega_sum  # Shape: (batch_size, proto_num, 1)
        return tf.concat([inputs, mass_omega_sum], axis=-1)  # Shape: (batch_size, proto_num, num_class + 1)

    def get_config(self):
        config = super(DS2_omega, self).get_config()
        config.update({'input_dim': self.input_dim, 'num_class': self.num_class})
        return config
    
@tf.keras.utils.register_keras_serializable()
class DS3_Dempster(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class, **kwargs):
        super(DS3_Dempster, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.num_class = num_class

    def call(self, inputs):
        # Expecting inputs of shape (batch_size, proto_num, num_class + 1)
        m1 = inputs[:, 0, :]
        omega1 = tf.expand_dims(inputs[:, 0, -1], -1)

        for i in range(1, self.input_dim):
            m2 = inputs[:, i, :]
            omega2 = tf.expand_dims(inputs[:, i, -1], -1)

            combine1 = m1 * m2
            combine2 = m1 * omega2
            combine3 = omega1 * m2

            combine2_3 = combine1 + combine2 + combine3

            normalization_factor = tf.reduce_sum(combine2_3, axis=-1, keepdims=True) + 1e-7
            combine2_3 = combine2_3 / normalization_factor

            m1 = combine2_3
            omega1 = tf.expand_dims(combine2_3[:, -1], -1)

        return m1

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_class + 1)

    def get_config(self):
        config = super(DS3_Dempster, self).get_config()
        config.update({'input_dim': self.input_dim, 'num_class': self.num_class})
        return config

@tf.keras.utils.register_keras_serializable()
class DS3_normalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DS3_normalize, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute the sum of inputs along the last axis and keep dimensions for broadcasting
        sum_inputs = tf.reduce_sum(inputs, axis=-1, keepdims=True)
        # Normalize the inputs by dividing by the sum plus a small epsilon for numerical stability
        normalized_inputs = inputs / (sum_inputs + 1e-7)
        return normalized_inputs

    def get_config(self):
        config = super(DS3_normalize, self).get_config()
        return config