import tensorflow as tf


def generator(input, name, training=True, reuse=False):

    with tf.variable_scope(name, reuse=reuse):

        x = tf.layers.conv2d(input, 32, 5, 2, padding='same')
        x = tf.maximum(0., x)

        x = tf.layers.conv2d(x, 128, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(0., x)

        x = tf.layers.conv2d(x, 256, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(0., x)

        # ---------------------------------------------------------------

        x = tf.layers.conv2d_transpose(x, 256, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(0., x)

        x = tf.layers.conv2d_transpose(x, 128, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(0., x)

        #out_mask = tf.layers.conv2d_transpose(x, 1, 5, 2, padding='same')
        #out_mask = tf.sigmoid(out_mask)
        out_image = tf.layers.conv2d_transpose(x, 3, 5, 2, padding='same')
        out_image = tf.tanh(out_image)

        #out = input * (1. - out_mask) + out_image * out_mask
        return out_image


def discriminator(x, name, reuse=False, alpha=0.2, training=True):

    with tf.variable_scope(name, reuse=reuse):

        x = tf.layers.conv2d(x, 32, 5, 2, padding='same')
        x = tf.maximum(alpha*x, x)

        x = tf.layers.conv2d(x, 64, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(alpha*x, x)

        x = tf.layers.conv2d(x, 128, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(alpha*x, x)

        x = tf.layers.conv2d(x, 256, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(alpha*x, x)

        flatten = tf.reshape(x, (-1, 4*4*256))
        logits = tf.layers.dense(flatten, 1)
        return logits
