import os
import cv2
import tensorflow as tf
import numpy as np
from Networks import discriminator, generator

print("Load Data")
IMAGES_PATH = "C:/data/CASIA-WebFace-Aligned"
images = []
for user_directory_name in os.listdir(IMAGES_PATH):
    if len(images) > 50000: break
    dir_path = os.path.join(IMAGES_PATH, user_directory_name)
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        images.append(img)
images = np.array(images)

input_real = tf.placeholder(tf.float32, (None, 64, 64, 3), name='input_real')
input_noise = tf.placeholder(tf.float32, (None, 100), name='input_noise')

gen_noise = generator(input_noise)
dis_logits_real = discriminator(input_real)
dis_logits_fake = discriminator(gen_noise, reuse=True)

dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_real, labels=tf.ones_like(dis_logits_real)))
dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_fake, labels=tf.zeros_like(dis_logits_real)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits_fake, labels=tf.ones_like(dis_logits_real)))
dis_loss = dis_loss_real + dis_loss_fake

# defining optimizers
total_vars = tf.trainable_variables()
dis_vars = [var for var in total_vars if var.name[0] == 'd']
gen_vars = [var for var in total_vars if var.name[0] == 'g']
dis_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(dis_loss, var_list=dis_vars)
gen_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(gen_loss, var_list=gen_vars)

batch_size = 128
iters = len(images)//batch_size
epochs = 100000
SAVE_PATH = 'C:/Users/Xiaomi/Pictures/test'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for i in range(iters-1):

            batch_images = images[i*batch_size:(i+1)*batch_size]
            batch_images = batch_images / 255.
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, 100))

            discriminator_loss, _ = sess.run([dis_loss, dis_opt], feed_dict={input_real: batch_images, input_noise: batch_noise})
            print("Discriminator Loss:", discriminator_loss)
            generated_image, generator_loss, _ = sess.run([gen_noise, gen_loss, gen_opt], feed_dict={input_real: batch_images, input_noise: batch_noise})
            print("Generator Loss:", generator_loss)

            if i % 10 == 0:
                print("Epoch {}/{}...".format(e+1, epochs), "Batch No {}/{}".format(i+1, iters))
                save_img = (generated_image[0] * 255.).astype(np.uint8)
                save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(SAVE_PATH, "example" + str(i % 100) + ".jpg"), save_img)
