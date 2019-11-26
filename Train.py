import os
import cv2
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from FaceAligment import preprocess_face
from Networks3 import discriminator, generator
import pandas as pd

IMG_SIZE = (128, 128)
CELEBA_LANDMARKS_PATH = "C:\data\celeba/list_landmarks_align_celeba.csv"
CELEBA_ATTRIBUTES_PATH = "C:\data\celeba/list_attr_celeba.csv"
CELEBA_IMAGES_PATH = "C:\data\celeba\img_align_celeba"

landmarks = pd.read_csv(CELEBA_LANDMARKS_PATH)
attributes = pd.read_csv(CELEBA_ATTRIBUTES_PATH)
df = pd.concat([landmarks.set_index('image_id'), attributes.set_index('image_id')], axis=1, join='inner').reset_index()
df["path"] = df["image_id"].apply(lambda x: os.path.join(CELEBA_IMAGES_PATH, x))
male_df = df[df["Male"] == 1]
female_df = df[df["Male"] == -1]

def get_landmarks(df):
    return df[[
        "nose_x", "nose_y",
        "leftmouth_x", "leftmouth_y",
        "rightmouth_x", "rightmouth_y",
        "lefteye_x", "lefteye_y",
        "righteye_x", "righteye_y"
    ]].to_numpy(dtype=np.float32).reshape((-1, 5, 2))

IMAGES_PATHS_A = np.array(male_df["path"].tolist())
IMAGES_PATHS_B = np.array(female_df["path"].tolist())
LANDMARKS_A = get_landmarks(male_df)
LANDMARKS_B = get_landmarks(female_df)

# Placeholders
A_image_placeholder = tf.placeholder(tf.float32, (None, IMG_SIZE[0], IMG_SIZE[1], 3), name='input_real_A')
B_image_placeholder = tf.placeholder(tf.float32, (None, IMG_SIZE[0], IMG_SIZE[1], 3), name='input_real_B')
A_stages_placeholder = [
    tf.placeholder(tf.float32, (None, IMG_SIZE[0]/2, IMG_SIZE[1]/2, 3), name='stage_2_A'),
    tf.placeholder(tf.float32, (None, IMG_SIZE[0]/4, IMG_SIZE[1]/4, 3), name='stage_4_A'),
    tf.placeholder(tf.float32, (None, IMG_SIZE[0]/8, IMG_SIZE[1]/8, 3), name='stage_8_A'),
    tf.placeholder(tf.float32, (None, IMG_SIZE[0]/16, IMG_SIZE[1]/16, 3), name='stage_16_A')
]
B_stages_placeholder = [
    tf.placeholder(tf.float32, (None, IMG_SIZE[0]/2, IMG_SIZE[1]/2, 3), name='stage_2_B'),
    tf.placeholder(tf.float32, (None, IMG_SIZE[0]/4, IMG_SIZE[1]/4, 3), name='stage_4_B'),
    tf.placeholder(tf.float32, (None, IMG_SIZE[0]/8, IMG_SIZE[1]/8, 3), name='stage_8_B'),
    tf.placeholder(tf.float32, (None, IMG_SIZE[0]/16, IMG_SIZE[1]/16, 3), name='stage_16_B')
]

#Network
generator_A_output_op, stagesA = generator(A_image_placeholder, "generator_A")
generator_B_output_op, stagesB  = generator(B_image_placeholder, "generator_B")
generator_ABA_op, _ = generator(generator_A_output_op, "generator_B", reuse=True)
generator_BAB_op, _ = generator(generator_B_output_op, "generator_A", reuse=True)
generator_A_identity_op, _ = generator(B_image_placeholder, "generator_A", reuse=True)
generator_B_identity_op, _ = generator(A_image_placeholder, "generator_B", reuse=True)


print(generator_B_output_op.get_shape())
print(generator_A_output_op.get_shape())
print(stagesB[0].get_shape(),stagesB[1].get_shape(),stagesB[2].get_shape(),stagesB[3].get_shape())
print(stagesA[0].get_shape(),stagesA[1].get_shape(),stagesA[2].get_shape(),stagesA[3].get_shape())
discriminator_A_fake_op = discriminator(generator_A_output_op, stagesA, "discriminator_A")
discriminator_B_fake_op = discriminator(generator_B_output_op, stagesB, "discriminator_B")
discriminator_A_real_op = discriminator(B_image_placeholder, B_stages_placeholder, "discriminator_A", reuse=True)
discriminator_B_real_op = discriminator(A_image_placeholder, A_stages_placeholder, "discriminator_B", reuse=True)

#LOSS GENERATOR
SMOOTH = 0.9
ALPHA_CYCLE = 10.
generator_cycle_loss_aba = tf.reduce_mean(tf.abs(generator_ABA_op - A_image_placeholder))
generator_cycle_loss_bab = tf.reduce_mean(tf.abs(generator_BAB_op - B_image_placeholder))
generator_A_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_A_fake_op, labels=tf.ones_like(discriminator_A_fake_op) * SMOOTH))
generator_B_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_B_fake_op, labels=tf.ones_like(discriminator_B_fake_op) * SMOOTH))
generator_dis_loss = generator_A_dis_loss + generator_B_dis_loss
generator_cycle_loss = ALPHA_CYCLE * (generator_cycle_loss_aba + generator_cycle_loss_bab)
generator_identity_loss = tf.reduce_mean(tf.abs(generator_A_identity_op - B_image_placeholder)) + tf.reduce_mean(tf.abs(generator_B_identity_op - A_image_placeholder))
generator_loss_total = generator_dis_loss + generator_cycle_loss + generator_identity_loss

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


def random_batch(array, landmarks, batch_size):
    range = np.arange(len(array))
    np.random.shuffle(range)
    indexes = range[:batch_size]
    paths = array[indexes]
    lnd = landmarks[indexes]
    data = []
    for i, l in zip(paths, lnd):
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_face(img, l, IMG_SIZE)
        data.append(img)
    return np.array(data)

BATCH_SIZE = 4
EPOCHS = 100000
SAVE_PATH ='C:/Users/Xiaomi/Pictures/test'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(EPOCHS):
        print("Epoch:", e)
        for iterate in range(10_000):
            #fig, ax = plt.subplots(1)
            #ax.imshow(random_batch(IMAGES_PATHS_B, LANDMARKS_B, BATCH_SIZE)[0])
            #plt.show()
            #Train Discriminator
            batch_a = random_batch(IMAGES_PATHS_A, LANDMARKS_A, BATCH_SIZE) / 127.5 - 1.
            batch_b = random_batch(IMAGES_PATHS_B, LANDMARKS_B, BATCH_SIZE) / 127.5 - 1.
            feed_dict = {
                A_image_placeholder: batch_a,
                B_image_placeholder: batch_b}
            for i in range(4):
                size = int(128 / 2**(i+1))
                ba = np.array([cv2.resize(img, (size, size)) for img in batch_a])
                bb = np.array([cv2.resize(img, (size, size)) for img in batch_b])
                feed_dict[A_stages_placeholder[i]] = ba / 127.5 - 1.
                feed_dict[B_stages_placeholder[i]] = bb / 127.5 - 1.
            discriminator_loss, _ = sess.run([discriminator_loss_total, dis_opt], feed_dict)
            print("Discriminator Loss:", discriminator_loss)

            #Train Generator
            batch_a = random_batch(IMAGES_PATHS_A, LANDMARKS_A, BATCH_SIZE) / 127.5 - 1.
            batch_b = random_batch(IMAGES_PATHS_B, LANDMARKS_B,  BATCH_SIZE) / 127.5 - 1.
            feed_dict = {
                A_image_placeholder: batch_a,
                B_image_placeholder: batch_b}
            for i in range(4):
                size = int(128 / 2**(i+1))
                ba = np.array([cv2.resize(img, (size, size)) for img in batch_a])
                bb = np.array([cv2.resize(img, (size, size)) for img in batch_b])
                feed_dict[A_stages_placeholder[i]] = ba / 127.5 - 1.
                feed_dict[B_stages_placeholder[i]] = bb / 127.5 - 1.

            generated_A_image, generated_B_image, generator_loss, dis_l, cycle_l, identity_l, _ = sess.run([generator_A_output_op,
                                                           generator_B_output_op,
                                                           generator_loss_total,
                                                           generator_dis_loss,
                                                           generator_cycle_loss,
                                                           generator_identity_loss,
                                                           gen_opt], feed_dict)
            print("Generator Loss:", generator_loss, "DIS LOSS:", dis_l, "CYCLE LOSS", cycle_l, "IDENTITY LOSS", identity_l)

            if iterate % 10 == 0:
                im = np.concatenate([batch_a[0], generated_A_image[0]], axis=1)
                save_img = ((im + 1.) * 127.5).astype(np.uint8)
                save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(SAVE_PATH, "example" + str(iterate % 100000) + ".jpg"), save_img)
