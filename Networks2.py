import tensorflow as tf

def feature_extractor(input, filters, is_training):
        layer = tf.layers.conv2d(input, filters=filters, kernel_size=3, strides=2, padding='same', use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=is_training)
        layer = tf.nn.relu(layer)
        return layer

def inverted_residual(input, input_filters, output_filters, stride, is_training):

        layer = tf.layers.conv2d(input, filters=input_filters, kernel_size=1, strides=1, padding='same', use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=is_training)
        layer = tf.nn.relu(layer)

        layer = tf.layers.separable_conv2d(layer, filters=input_filters, kernel_size=3, strides=stride, padding='same', use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=is_training)
        layer = tf.nn.relu(layer)

        layer = tf.layers.conv2d(layer, filters=output_filters, kernel_size=1, strides=1, padding='same', use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=is_training)
        if stride == 1 and input_filters == output_filters:
            layer = layer + input
        return layer


def residual_block(input, in_filters, out_filters, is_training):
    layer = tf.layers.separable_conv2d(input, filters=in_filters, kernel_size=3, strides=1, padding='same', use_bias=False)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    layer = tf.layers.conv2d(layer, filters=out_filters, kernel_size=1, strides=1, padding='same', use_bias=False)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)

    layer = tf.layers.separable_conv2d(layer, filters=out_filters, kernel_size=3, strides=1, padding='same', use_bias=False)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    layer = tf.layers.conv2d(layer, filters=out_filters, kernel_size=1, strides=1, padding='same', use_bias=False)
    layer = tf.layers.batch_normalization(layer, training=is_training)

    residual = tf.layers.conv2d(input, filters=out_filters, kernel_size=1, strides=1, padding='same', use_bias=False)
    residual = tf.layers.batch_normalization(residual, training=is_training)
    return tf.nn.relu(layer + residual)


def generator(input, name, training=True, reuse=False):

    with tf.variable_scope(name, reuse=reuse):
        input_features = feature_extractor(input, 32, training)  # 1/2

        stage1 = inverted_residual(input_features, 32, 16, 1, training)

        stage2 = inverted_residual(stage1, 16, 24, 2, training)  # 1/4
        stage2 = inverted_residual(stage2, 24, 24, 1, training)

        stage3 = inverted_residual(stage2, 24, 32, 2, training)  # 1/8
        stage3 = inverted_residual(stage3, 32, 32, 1, training)
        stage3 = inverted_residual(stage3, 32, 32, 1, training)

        #stage4 = inverted_residual(stage3, 32, 64, 2, training)  # 1/16
        #stage4 = inverted_residual(stage4, 64, 64, 1, training)
        #stage4 = inverted_residual(stage4, 64, 64, 1, training)
        #stage4 = inverted_residual(stage4, 64, 64, 1, training)
        #stage4 = inverted_residual(stage4, 64, 96, 1, training)
        #stage4 = inverted_residual(stage4, 96, 96, 1, training)
        #stage4 = inverted_residual(stage4, 96, 96, 1, training)
#
        #stage5 = inverted_residual(stage4, 64, 160, 2, training)  # 1/32
        #stage5 = inverted_residual(stage5, 160, 160, 1, training)
        #stage5 = inverted_residual(stage5, 160, 160, 1, training)
        #stage5 = inverted_residual(stage5, 160, 320, 1, training)

        #stage_5_up = residual_block(stage5, 320, 96, training)
        #stage_5_up = tf.layers.conv2d_transpose(stage_5_up, 96, kernel_size=4, strides=2, use_bias=False, padding="same")
#
        #stage_4_up = residual_block(stage_5_up + stage4, 96, 32, training)
        #stage_4_up = tf.layers.conv2d_transpose(stage_4_up, 32, kernel_size=4, strides=2, use_bias=False, padding="same")

        stage_3_up = residual_block(stage3, 32, 24, training) #stage_4_up + stage3
        stage_3_up = tf.layers.conv2d_transpose(stage_3_up, 24, kernel_size=4, strides=2, use_bias=False, padding="same")

        stage_2_up = residual_block(stage_3_up + stage2, 24, 16, training)
        stage_2_up = tf.layers.conv2d_transpose(stage_2_up, 16, kernel_size=4, strides=2, use_bias=False, padding="same")

        stage_1_up = residual_block(stage_2_up, 16, 8, training)
        stage_1_up = tf.layers.conv2d_transpose(stage_1_up, 8, kernel_size=4, strides=2, use_bias=False, padding="same")

        result = tf.layers.conv2d(stage_1_up, 3, kernel_size=1, strides=1, use_bias=False, padding="same")
        result = tf.tanh(result)
        mask = tf.layers.conv2d(stage_1_up, 3, kernel_size=1, strides=1, use_bias=False, padding="same")
        mask = tf.sigmoid(mask)

        return input * (1. - mask) + result * mask


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
