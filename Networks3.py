import tensorflow as tf


def generator(input, name, training=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        #DOWN 1
        stage1 = tf.layers.conv2d(input, filters=16, kernel_size=3, strides=1, padding='same')
        stage1 = tf.layers.batch_normalization(stage1, training=training)
        stage1 = tf.nn.leaky_relu(stage1)
        stage1 = tf.layers.conv2d(stage1, filters=16, kernel_size=3, strides=2, padding='same')
        stage1 = tf.layers.batch_normalization(stage1, training=training)
        stage1 = tf.nn.leaky_relu(stage1)

        #DOWN 2
        stage2 = tf.layers.conv2d(stage1, filters=32, kernel_size=3, strides=1, padding='same')
        stage2 = tf.layers.batch_normalization(stage2, training=training)
        stage2 = tf.nn.leaky_relu(stage2)
        stage2 = tf.layers.conv2d(stage2, filters=32, kernel_size=3, strides=2, padding='same')
        stage2 = tf.layers.batch_normalization(stage2, training=training)
        stage2 = tf.nn.leaky_relu(stage2)

        #DOWN 3
        stage3 = tf.layers.conv2d(stage2, filters=64, kernel_size=3, strides=1, padding='same')
        stage3 = tf.layers.batch_normalization(stage3, training=training)
        stage3 = tf.nn.leaky_relu(stage3)
        stage3 = tf.layers.conv2d(stage3, filters=64, kernel_size=3, strides=2, padding='same')
        stage3 = tf.layers.batch_normalization(stage3, training=training)
        stage3 = tf.nn.leaky_relu(stage3)

        #DOWN 4
        stage4 = tf.layers.conv2d(stage3, filters=64, kernel_size=3, strides=1, padding='same')
        stage4 = tf.layers.batch_normalization(stage4, training=training)
        stage4 = tf.nn.leaky_relu(stage4)
        stage4 = tf.layers.conv2d(stage4, filters=64, kernel_size=3, strides=2, padding='same')
        stage4 = tf.layers.batch_normalization(stage4, training=training)
        stage4 = tf.nn.leaky_relu(stage4)

        stage4_up = tf.layers.conv2d(stage4, filters=64, kernel_size=3, strides=1, padding='same')
        stage4_up = tf.layers.batch_normalization(stage4_up, training=training)
        stage4_up = tf.nn.leaky_relu(stage4_up)

        #UP4
        stage3_up = tf.layers.conv2d_transpose(stage4_up, filters=64, kernel_size=3, strides=2, padding='same')
        stage3_up = tf.layers.batch_normalization(stage3_up, training=training)
        stage3_up = tf.nn.leaky_relu(stage3_up)
        stage3_up = tf.concat([stage3_up, stage3], axis=-1)
        stage3_up = tf.layers.conv2d(stage3_up, filters=64, kernel_size=3, strides=1, padding='same')
        stage3_up = tf.layers.batch_normalization(stage3_up, training=training)
        stage3_up = tf.nn.leaky_relu(stage3_up)

        #UP3
        stage2_up = tf.layers.conv2d_transpose(stage3_up, filters=64, kernel_size=3, strides=2, padding='same')
        stage2_up = tf.layers.batch_normalization(stage2_up, training=training)
        stage2_up = tf.nn.leaky_relu(stage2_up)
        stage2_up = tf.concat([stage2_up, stage2], axis=-1)
        stage2_up = tf.layers.conv2d(stage2_up, filters=64, kernel_size=3, strides=1, padding='same')
        stage2_up = tf.layers.batch_normalization(stage2_up, training=training)
        stage2_up = tf.nn.leaky_relu(stage2_up)

        #UP2
        stage1_up = tf.layers.conv2d_transpose(stage2_up, filters=32, kernel_size=3, strides=2, padding='same')
        stage1_up = tf.layers.batch_normalization(stage1_up, training=training)
        stage1_up = tf.nn.leaky_relu(stage1_up)
        stage1_up = tf.concat([stage1_up, stage1], axis=-1)
        stage1_up = tf.layers.conv2d(stage1_up, filters=32, kernel_size=3, strides=1, padding='same')
        stage1_up = tf.layers.batch_normalization(stage1_up, training=training)
        stage1_up = tf.nn.leaky_relu(stage1_up)

        #UP1
        result = tf.layers.conv2d_transpose(stage1_up, filters=16, kernel_size=3, strides=2, padding='same')
        result = tf.layers.batch_normalization(result, training=training)
        result = tf.nn.leaky_relu(result)
        result = tf.layers.conv2d(result, filters=16, kernel_size=3, strides=1, padding='same')
        result = tf.layers.batch_normalization(result, training=training)
        result = tf.nn.leaky_relu(result)

        result = tf.layers.conv2d(result, 3, kernel_size=1, strides=1, use_bias=False, padding="same")
        result = tf.tanh(result)

        return result, [
            tf.layers.conv2d(stage1_up, 3, kernel_size=1, strides=1, activation=tf.nn.tanh),
            tf.layers.conv2d(stage2_up, 3, kernel_size=1, strides=1, activation=tf.nn.tanh),
            tf.layers.conv2d(stage3_up, 3, kernel_size=1, strides=1, activation=tf.nn.tanh),
            tf.layers.conv2d(stage4_up, 3, kernel_size=1, strides=1, activation=tf.nn.tanh) ]


def discriminator(input, stages, name, reuse=False, training=True):
    with tf.variable_scope(name, reuse=reuse):
        layer = tf.layers.conv2d(input, filters=16, kernel_size=3, strides=1, padding='same')
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.conv2d(layer, filters=16, kernel_size=3, strides=2, padding='same')
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.nn.leaky_relu(layer)
        layer = tf.concat([layer, stages[0]], axis=-1)

        layer = tf.layers.conv2d(layer, filters=32, kernel_size=3, strides=1, padding='same')
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.conv2d(layer, filters=32, kernel_size=3, strides=2, padding='same')
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.nn.leaky_relu(layer)
        layer = tf.concat([layer, stages[1]], axis=-1)

        layer = tf.layers.conv2d(layer, filters=64, kernel_size=3, strides=1, padding='same')
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.conv2d(layer, filters=64, kernel_size=3, strides=2, padding='same')
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.nn.leaky_relu(layer)
        layer = tf.concat([layer, stages[2]], axis=-1)

        layer = tf.layers.conv2d(layer, filters=64, kernel_size=3, strides=1, padding='same')
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.conv2d(layer, filters=64, kernel_size=3, strides=2, padding='same')
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.nn.leaky_relu(layer)
        layer = tf.concat([layer, stages[3]], axis=-1)

        layer = tf.layers.conv2d(layer, filters=64, kernel_size=3, strides=1, padding='same')
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.nn.leaky_relu(layer)
        layer = tf.layers.conv2d(layer, filters=64, kernel_size=3, strides=2, padding='same')
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.nn.leaky_relu(layer)

        flatten = tf.layers.flatten(layer)
        logits = tf.layers.dense(flatten, 1)
        return logits
