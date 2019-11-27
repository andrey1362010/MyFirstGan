import tensorflow as tf

# Pixelwise feature vector normalization.
def pixel_normalization(input, epsilon=1e-8):
    return input * tf.rsqrt(tf.reduce_mean(tf.square(input), axis=1, keepdims=True) + epsilon)

# Minibatch standard deviation.
def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

def generator_conv(input, filters, stride):
    layer = tf.layers.conv2d(input, filters=filters, kernel_size=3, strides=stride, padding='same')
    layer = pixel_normalization(layer)
    layer = tf.nn.leaky_relu(layer)
    return layer

def downscale(input):
    return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def upscale(input):
    wh = [2 * tf.shape(input)[1], 2 * tf.shape(input)[2]]
    return tf.image.resize_images(input, wh, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

def generator(input, name, training=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        stage1 = generator_conv(input, 16, stride=1)
        stage1 = generator_conv(stage1, 16, stride=1)

        stage2 = downscale(stage1)
        stage2 = generator_conv(stage2, 32, stride=1)
        stage2 = generator_conv(stage2, 32, stride=1)

        stage3 = downscale(stage2)
        stage3 = generator_conv(stage3, 64, stride=1)
        stage3 = generator_conv(stage3, 64, stride=1)

        stage4 = downscale(stage3)
        stage4 = generator_conv(stage4, 128, stride=1)
        stage4 = generator_conv(stage4, 128, stride=1)

        stage5 = downscale(stage4)
        stage5 = generator_conv(stage5, 128, stride=1)
        stage5 = generator_conv(stage5, 128, stride=1)

        stage6 = downscale(stage5)
        stage6 = generator_conv(stage6, 128, stride=1)
        stage6 = generator_conv(stage6, 128, stride=1)

        assert (stage6.get_shape()[1] == 4)
        #-----------------------------

        stage6_up = generator_conv(stage6, 128, stride=1)
        stage6_up = generator_conv(stage6_up, 128, stride=1)

        stage5_up = upscale(stage6_up)
        stage5_up = tf.concat([stage5_up, stage5], axis=-1)
        stage5_up = generator_conv(stage5_up, 128, stride=1)
        stage5_up = generator_conv(stage5_up, 128, stride=1)

        stage4_up = upscale(stage5_up)
        stage4_up = tf.concat([stage4_up, stage4], axis=-1)
        stage4_up = generator_conv(stage4_up, 128, stride=1)
        stage4_up = generator_conv(stage4_up, 128, stride=1)

        stage3_up = upscale(stage4_up)
        stage3_up = tf.concat([stage3_up, stage3], axis=-1)
        stage3_up = generator_conv(stage3_up, 64, stride=1)
        stage3_up = generator_conv(stage3_up, 64, stride=1)

        stage2_up = upscale(stage3_up)
        stage2_up = tf.concat([stage2_up, stage2], axis=-1)
        stage2_up = generator_conv(stage2_up, 32, stride=1)
        stage2_up = generator_conv(stage2_up, 32, stride=1)

        stage1_up = upscale(stage2_up)
        stage1_up = tf.concat([stage1_up, stage1], axis=-1)
        stage1_up = generator_conv(stage1_up, 16, stride=1)
        stage1_up = generator_conv(stage1_up, 16, stride=1)

        result = tf.layers.conv2d(stage1_up, 3, kernel_size=1, strides=1, use_bias=False, padding="same")
        result = tf.tanh(result)

        return result, [
            tf.layers.conv2d(stage2_up, 3, kernel_size=1, strides=1, activation=tf.nn.tanh),
            tf.layers.conv2d(stage3_up, 3, kernel_size=1, strides=1, activation=tf.nn.tanh),
            tf.layers.conv2d(stage4_up, 3, kernel_size=1, strides=1, activation=tf.nn.tanh),
            tf.layers.conv2d(stage5_up, 3, kernel_size=1, strides=1, activation=tf.nn.tanh),
            tf.layers.conv2d(stage6_up, 3, kernel_size=1, strides=1, activation=tf.nn.tanh)]


def discriminator_conv(input, filters, kernel=3, stride=1):
    layer = tf.layers.conv2d(input, filters=filters, kernel_size=kernel, strides=stride, padding='same')
    layer = tf.nn.leaky_relu(layer)
    return layer

def discriminator(input, stages, name, reuse=False, training=True):
    with tf.variable_scope(name, reuse=reuse):
        layer = discriminator_conv(input, filters=16, stride=1)
        layer = discriminator_conv(layer, filters=16, stride=1)
        layer = downscale(layer)

        second_input = tf.layers.conv2d(stages[0], filters=16, kernel_size=1, strides=1, padding='same')
        layer = tf.concat([layer, second_input], axis=-1)
        layer = discriminator_conv(layer, filters=32, stride=1)
        layer = discriminator_conv(layer, filters=32, stride=1)
        layer = downscale(layer)

        second_input = tf.layers.conv2d(stages[1], filters=32, kernel_size=1, strides=1, padding='same')
        layer = tf.concat([layer, second_input], axis=-1)
        layer = discriminator_conv(layer, filters=64, stride=1)
        layer = discriminator_conv(layer, filters=64, stride=1)
        layer = downscale(layer)

        second_input = tf.layers.conv2d(stages[2], filters=64, kernel_size=1, strides=1, padding='same')
        layer = tf.concat([layer, second_input], axis=-1)
        layer = discriminator_conv(layer, filters=128, stride=1)
        layer = discriminator_conv(layer, filters=128, stride=1)
        layer = downscale(layer)

        second_input = tf.layers.conv2d(stages[3], filters=128, kernel_size=1, strides=1, padding='same')
        layer = tf.concat([layer, second_input], axis=-1)
        layer = discriminator_conv(layer, filters=128, stride=1)
        layer = discriminator_conv(layer, filters=128, stride=1)
        layer = downscale(layer)

        second_input = tf.layers.conv2d(stages[4], filters=128, kernel_size=1, strides=1, padding='same')
        layer = tf.concat([layer, second_input], axis=-1)
        layer = discriminator_conv(layer, filters=128, stride=1)
        layer = discriminator_conv(layer, filters=128, stride=1)

        layer = minibatch_stddev_layer(layer, 4)
        layer = discriminator_conv(layer, filters=64, kernel=3, stride=1)
        layer = discriminator_conv(layer, filters=32, kernel=4, stride=1)
        flatten = tf.layers.flatten(layer)
        logits = tf.layers.dense(flatten, 1)
        return logits
