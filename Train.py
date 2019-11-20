import os
import cv2
import tensorflow as tf
import numpy as np
from Networks2 import discriminator, generator

IMG_SIZE = (128, 128)

#for dir in os.listdir("/media/andrey/ssdbig1/data/ms1m_aligned_v2"):
#    dir = os.path.join("/media/andrey/ssdbig1/data/ms1m_aligned_v2", dir)
#    print(dir, len(os.listdir(dir)))

IMAGES_A_PATH = "/media/andrey/ssdbig1/data/ms1m_aligned_v2/m.030x8c" #MEN
IMAGES_B_PATH = "/media/andrey/ssdbig1/data/ms1m_aligned_v2/m.0b6hb7j" #WOMEN
def read_images(path):
    images = []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
    return np.array(images)
IMAGES_A = read_images(IMAGES_A_PATH)
IMAGES_B = read_images(IMAGES_B_PATH)

# Placeholders
A_image_placeholder = tf.placeholder(tf.float32, (None, IMG_SIZE[0], IMG_SIZE[1], 3), name='input_real')
B_image_placeholder = tf.placeholder(tf.float32, (None, IMG_SIZE[0], IMG_SIZE[1], 3), name='input_real')

#Network
generator_A_output_op = generator(A_image_placeholder, "generator_A")
generator_B_output_op = generator(B_image_placeholder, "generator_B")
generator_ABA_op = generator(generator(A_image_placeholder, "generator_A", reuse=True), "generator_B", reuse=True)
generator_BAB_op = generator(generator(B_image_placeholder, "generator_B", reuse=True), "generator_A", reuse=True)

discriminator_A_fake_op = discriminator(generator_A_output_op, "discriminator_A")
discriminator_B_fake_op = discriminator(generator_B_output_op, "discriminator_B")
discriminator_A_real_op = discriminator(B_image_placeholder, "discriminator_A", reuse=True)
discriminator_B_real_op = discriminator(A_image_placeholder, "discriminator_B", reuse=True)

#LOSS GENERATOR
SMOOTH = 0.9
ALPHA_CYCLE = 10.
generator_cycle_loss_aba = tf.reduce_mean(tf.abs(generator_ABA_op - A_image_placeholder))
generator_cycle_loss_bab = tf.reduce_mean(tf.abs(generator_BAB_op - B_image_placeholder))
generator_A_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_A_fake_op, labels=tf.ones_like(discriminator_A_fake_op) * SMOOTH))
generator_B_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_B_fake_op, labels=tf.ones_like(discriminator_B_fake_op) * SMOOTH))
generator_dis_loss = generator_A_dis_loss + generator_B_dis_loss
generator_cycle_loss = ALPHA_CYCLE * (generator_cycle_loss_aba + generator_cycle_loss_bab)
generator_loss_total = generator_dis_loss + generator_cycle_loss

#LOSS DISCRIMINATOR
discriminator_A_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_A_fake_op, labels=tf.zeros_like(discriminator_A_fake_op)))
discriminator_B_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_B_fake_op, labels=tf.zeros_like(discriminator_B_fake_op)))
discriminator_A_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_A_real_op, labels=tf.ones_like(discriminator_A_real_op) * SMOOTH))
discriminator_B_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_B_real_op, labels=tf.ones_like(discriminator_B_real_op) * SMOOTH))
discriminator_loss_total = discriminator_A_fake_loss + discriminator_B_fake_loss + discriminator_A_real_loss + discriminator_B_real_loss

# OPTIMIZERS
total_vars = tf.trainable_variables()
dis_vars = [var for var in total_vars if var.name[0] == 'd']
gen_vars = [var for var in total_vars if var.name[0] == 'g']
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    dis_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(discriminator_loss_total, var_list=dis_vars)
    gen_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(generator_loss_total, var_list=gen_vars)


def random_batch(array, batch_size):
    range = np.arange(len(array))
    np.random.shuffle(range)
    indexes = range[:batch_size]
    return array[indexes]

BATCH_SIZE = 4
EPOCHS = 100000
SAVE_PATH = '/media/andrey/ssdbig1/data/tmp'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(EPOCHS):
        print("Epoch:", e)
        for i in range(10_000):

            #Train Discriminator
            batch_a = random_batch(IMAGES_A, BATCH_SIZE) / 255.
            batch_b = random_batch(IMAGES_B, BATCH_SIZE) / 255.
            discriminator_loss, _ = sess.run([discriminator_loss_total, dis_opt], feed_dict={
                A_image_placeholder: batch_a,
                B_image_placeholder: batch_b})
            print("Discriminator Loss:", discriminator_loss)

            #Train Generator
            batch_a = random_batch(IMAGES_A, BATCH_SIZE) / 255.
            batch_b = random_batch(IMAGES_B, BATCH_SIZE) / 255.
            generated_A_image, generated_B_image, generator_loss, dis_l, cycle_l, _ = sess.run([generator_A_output_op,
                                                           generator_B_output_op,
                                                           generator_loss_total,
                                                           generator_dis_loss,
                                                           generator_cycle_loss,
                                                           gen_opt], feed_dict={
                                                                    A_image_placeholder: batch_a,
                                                                    B_image_placeholder: batch_b})
            print("Generator Loss:", generator_loss, "DIS LOSS:", dis_l, "CYCLE LOSS", cycle_l)

            if i % 1000 == 0:
                im = np.concatenate([batch_a[0], generated_A_image[0]], axis=1)
                save_img = (im * 255.).astype(np.uint8)
                save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(SAVE_PATH, "example" + str(i % 100000) + ".jpg"), save_img)
