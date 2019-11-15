import os
import cv2
import tensorflow as tf
import numpy as np
from Networks import discriminator, generator

print("Load Data")
IMAGES_PATH = "C:/data/CASIA-WebFace-Aligned"
images = []
for user_directory_name in os.listdir(IMAGES_PATH):
    if len(images) > 100: break
    dir_path = os.path.join(IMAGES_PATH, user_directory_name)
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        images.append(img)


print("Placeholders")
noise_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name="noise")
img_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name="img")
result_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="result")
training_placeholder = tf.placeholder(dtype=tf.bool)

print("Network")
img_discriminator_op = discriminator(img_placeholder, training_placeholder, name="discriminator", reuse=False)
generator_op = generator(noise_placeholder, training_placeholder, name="generator")
generator_discriminator_op = discriminator(generator_op, training_placeholder, name="discriminator", reuse=True)


print("Generator variables")
trainable_generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
update_ops_generator = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="generator")

print("Discriminator variables")
trainable_discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
update_ops_discriminator = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="discriminator")

print("Loss [1 - real, 0 - fake]")
generator_loss = tf.reduce_mean(tf.keras.losses.MSE(1., generator_discriminator_op))
discriminator_loss = tf.reduce_mean(tf.keras.losses.MSE(result_placeholder, img_discriminator_op))

print("Training Settings")
generator_optimizer = tf.train.AdamOptimizer()
discriminator_optimizer = tf.train.AdamOptimizer()
with tf.control_dependencies(update_ops_generator):
    generator_train_step = generator_optimizer.minimize(generator_loss, var_list=trainable_generator_variables)
with tf.control_dependencies(update_ops_discriminator):
    discriminator_train_step = generator_optimizer.minimize(discriminator_loss, var_list=trainable_discriminator_variables)

SAVE_PATH = "C:/Users/Xiaomi/Pictures/test"
BATCH_SIZE = 3
sess = tf.Session()
sess.run(tf.global_variables_initializer())
counter = 0
for epoch_num in range(100):
    print("START EPOCH:", epoch_num)
    for index in range(0, len(images), BATCH_SIZE):

        noise_batch = np.random.rand(BATCH_SIZE, 64, 64, 1)
        generated_image, g_loss, _ = sess.run([generator_op, generator_loss, generator_train_step], feed_dict={noise_placeholder: noise_batch, training_placeholder: True})
        print("Generator Loss:", g_loss)

        batch = images[index: index + BATCH_SIZE]
        batch = [(img - 127.5) / 127.5 for img in batch]
        result_batch = [[1.] for i in range(BATCH_SIZE)]
        batch.extend(list(generated_image))
        result_batch.extend([[0.] for i in range(BATCH_SIZE)])
        noise_batch = np.zeros((2 * BATCH_SIZE, 64, 64, 1)) # Stub
        d_loss, _ = sess.run([discriminator_loss, discriminator_train_step], feed_dict={img_placeholder: batch, noise_placeholder: noise_batch, result_placeholder: result_batch, training_placeholder: True})
        print("Discriminator Loss:", d_loss)

        counter += 1
        if counter % 100 == 0:
            save_img = (generated_image[0] * 127.5 + 127.5).astype(np.uint8)
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(SAVE_PATH, "example" + str(counter % 1000) + ".jpg"), save_img)