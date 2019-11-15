import tensorflow as tf

def discriminator(image, is_training, name="discriminator", reuse=True):

    with tf.variable_scope(name, reuse=reuse):

        # Layer 1
        layer = tf.layers.conv2d(image, 32, kernel_size=4, strides=2, padding="same")
        layer = tf.nn.leaky_relu(layer)

        # Layer 2
        layer = tf.layers.conv2d(layer, 64, kernel_size=4, strides=2, padding="same")
        layer = tf.layers.batch_normalization(layer, training=is_training)
        layer = tf.nn.leaky_relu(layer)

        # Layer 3
        layer = tf.layers.conv2d(layer, 128, kernel_size=4, strides=2, padding="same")
        layer = tf.layers.batch_normalization(layer, training=is_training)
        layer = tf.nn.leaky_relu(layer)

        # Layer 4
        layer = tf.layers.conv2d(layer, 256, kernel_size=4, strides=2, padding="same")
        layer = tf.layers.batch_normalization(layer, training=is_training)
        layer = tf.nn.leaky_relu(layer)

        # Global Max Pooling
        result = tf.reduce_mean(layer, axis=[1, 2, 3])
        return result

def generator(noise, is_training, name="generator"):

    NUM_RES_BLOCKS = 6
    with tf.variable_scope(name):

        # Down Layer 1
        layer = tf.layers.conv2d(noise, 32, kernel_size=7, strides=1, padding="same")
        layer = tf.layers.batch_normalization(layer, training=is_training)
        layer = tf.nn.relu(layer)

        # Down Layer 2
        layer = tf.layers.conv2d(layer, 64, kernel_size=3, strides=2, padding="same")
        layer = tf.layers.batch_normalization(layer, training=is_training)
        layer = tf.nn.relu(layer)

        # Down Layer 3
        layer = tf.layers.conv2d(layer, 128, kernel_size=3, strides=2, padding="same")
        layer = tf.layers.batch_normalization(layer, training=is_training)
        layer = tf.nn.relu(layer)

        # Res Layers
        for block_id in range(NUM_RES_BLOCKS):
            start_layer = layer
            layer = tf.layers.conv2d(layer, 128, kernel_size=3, strides=1, padding="same")
            layer = tf.layers.batch_normalization(layer, training=is_training)
            layer = tf.nn.relu(layer)
            layer = tf.layers.conv2d(layer, 128, kernel_size=3, strides=1, padding="same")
            layer = tf.layers.batch_normalization(layer, training=is_training)
            layer = tf.nn.relu(start_layer + layer)

        # Up Layer 3
        layer = tf.layers.conv2d_transpose(layer, 64, kernel_size=3, strides=2, padding="same")
        layer = tf.layers.batch_normalization(layer, training=is_training)
        layer = tf.nn.relu(layer)

        # Up Layer 2
        layer = tf.layers.conv2d_transpose(layer, 32, kernel_size=3, strides=2, padding="same")
        layer = tf.layers.batch_normalization(layer, training=is_training)
        layer = tf.nn.relu(layer)

        # Up Layer 1
        layer = tf.layers.conv2d_transpose(layer, 3, kernel_size=7, strides=1, padding="same")
        layer = tf.layers.batch_normalization(layer, training=is_training)
        layer = tf.nn.relu(layer)
        return tf.nn.tanh(layer)

