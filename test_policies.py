from ncps.tf import CfC
import tensorflow as tf

class CfC_net(tf.keras.Model):
    def __init__(self, n_actions):
        super().__init__()
        self.rnn = CfC(64, return_sequences=True, return_state=True)
        self.linear = tf.keras.layers.Dense(n_actions)

    def get_initial_states(self, batch_size=1):
        return self.rnn.cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

    def call(self, x, training=None, **kwargs):
        has_hx = isinstance(x, list) or isinstance(x, tuple)
        initial_state = None
        if has_hx:
            # additional inputs are passed as a tuple
            x, initial_state = x
        x, next_state = self.rnn(x, initial_state=initial_state)
        x = self.linear(x)
        if has_hx:
            return (x, next_state)
        return x

def network_fn(X, n_actions):
    model = CfC_net(n_actions)
    return model(X)



