from tensorflow.keras.layers import Layer
import tensorflow as tf

class Layer1dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputEmbedding, validationEmbedding): #līdzības aprēķins
        return tf.math.abs(inputEmbedding - validationEmbedding)
