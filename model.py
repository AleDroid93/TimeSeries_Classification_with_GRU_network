import tensorflow as tf

class MyGRU(tf.keras.Model):
    def __init__(self, units, n_classes, dropout_rate = 0.0, hidden_activation='relu', output_activation='softmax',
                 name='gruNetwork',
                 **kwargs):
        # chiamata al costruttore della classe padre, Model
        super(MyGRU, self).__init__(name=name, **kwargs)
        # definizione dei layers del modello
        self.gru1 = tf.keras.layers.RNN(tf.keras.layers.GRUCell(units), return_sequences=True, return_state=True,
                                         name="gru1")
        self.gru2 = tf.keras.layers.RNN(tf.keras.layers.GRUCell(units), name="gru2")
        self.model_output = tf.keras.layers.Dense(units=n_classes, activation=output_activation)

    def call(self, inputs, training=False):
        # definisco il flusso, che la rete rappresentata dal modello, deve seguire.
        inputs = self.gru1(inputs, training=training)
        inputs = self.gru2(inputs, training=training)
        return self.model_output(inputs)