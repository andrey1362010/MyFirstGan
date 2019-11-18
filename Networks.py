import tensorflow as tf


def generator(noise, reuse=False, alpha=0.2, training=True):

    with tf.variable_scope('generator', reuse=reuse):

        x = tf.layers.dense(noise, 4*4*512)
        x = tf.reshape(x, (-1, 4, 4, 512))
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(0., x)

        x = tf.layers.conv2d_transpose(x, 256, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(0., x)

        x = tf.layers.conv2d_transpose(x, 128, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(0., x)

        x = tf.layers.conv2d_transpose(x, 64, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.maximum(0., x)

        logits = tf.layers.conv2d_transpose(x, 3, 5, 2, padding='same')
        out = tf.tanh(logits)

        return out


def discriminator(x, reuse=False, alpha=0.2, training=True):

    with tf.variable_scope('discriminator', reuse=reuse):

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
