import tensorflow as tf


def reset():
    """Clear graph and reset layer names."""
    tf.keras.backend.clear_session()
    tf.keras.backend.reset_uids()
