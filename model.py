from audioop import cross
from time import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Dense, Reshape, Activation, BatchNormalization, Conv2DTranspose, Conv2D, Flatten, Dropout, LeakyReLU
from matplotlib import pyplot as plt
import os
import random
tf.config.run_functions_eagerly(True)
EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16

BATCH_SIZE=256

def loadData():
    train = np.load('train.npy')
    return train

train = loadData()
train_dataset = tf.data.Dataset.from_tensor_slices(train).shuffle(60000).batch(128)

def defineGenerator():
    input = Input(shape=(100,))
    x = Dense(4*4*1024, use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Reshape((4,4,1024))(x)

    
    x = Conv2DTranspose(512,(5,5), strides=(2,2), padding='same', use_bias=False)(x)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    

    x = Conv2DTranspose(256,(5,5), strides=(2,2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    
    x = Conv2DTranspose(128,(5,5), strides=(2,2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(3,(5,5), strides=(2,2), padding='same', use_bias=False)(x)
    output = Activation('tanh')(x)
    model = Model(inputs=input, outputs=output)
    return model

generator = defineGenerator()

def defineDiscriminator():

    input = Input(shape=(64,64,3))
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    output = Dense(1)(x)
    model = Model(inputs=input, outputs=output)
    return model

discriminator = defineDiscriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminatorLoss(fake_output, real_output):
    # tf.zeros_like creates an array with the same shape but filled with all 0
    # the more 1 in the fake_output means a larger distance between the cross_entropy and so a larger loss
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    total_loss = 0.5 * (fake_loss+real_loss)
    return total_loss

def generatorLoss(fake_output):
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return fake_loss

generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)



@tf.function
def train_step(images, disc_losses, gen_losses):

    
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generatorLoss(fake_output)
      disc_loss = discriminatorLoss(real_output, fake_output)
      print(gen_loss.numpy())
      print(disc_loss.numpy())
      gen_losses.append(gen_loss.numpy())
      disc_losses.append(disc_loss.numpy())


    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(((predictions[i, :, :, :] + 1) / 2))
      plt.axis('off')

  plt.savefig('./Generated/image_at_epoch_{:04d}.png'.format(epoch))


def train(dataset, epochs):
  
  for epoch in range(epochs):
    gen_losses = []
    disc_losses = []
    start = time()

    for count, image_batch in enumerate(dataset):
        print(f'Epoch {epoch}, Batch {count}/{len(dataset)} Completed')
        train_step(image_batch, gen_losses, disc_losses)

    # Produce images for the GIF as you go
    generate_and_save_images(generator,
                          epoch + 1,
                          seed) 

    # Save the model every 15 epochs  
    if (epoch + 1) % 15 == 0:
      print('saved checkpoint')
      checkpoint.save(file_prefix = checkpoint_prefix)
    print(f'Generator Loss: {np.average(gen_losses)}, Discriminator Loss: {np.average(disc_losses)}')
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time()-start))
  
  # Generate after the final epoch
  generate_and_save_images(generator,
                           epochs,
                           seed)

train(train_dataset, EPOCHS)
