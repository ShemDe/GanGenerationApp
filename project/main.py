from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, ReLU, LeakyReLU, Dropout
from keras.utils import plot_model
from keras.optimizers import Adam
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
import cv2
import matplotlib
import matplotlib.pyplot as plt
import graphviz
import sys
import os

extract_dir = '*/bonsai.zip'
if not os.path.isdir(extract_dir):
    import zipfile
    zip_path = '*/bonsai.zip'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

CATEGORIES = set(["bonsai"])

ImagePaths=[]
for category in CATEGORIES:
    for image in list(os.listdir(extract_dir+"/"+category)):
        ImagePaths=ImagePaths+[extract_dir+"/"+category+"/"+image]

# Load images and resize to 64 x 64
data_lowres = []
for img in ImagePaths:
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lowres = cv2.resize(image, (64, 64))
    data_lowres.append(image_lowres)

# Convert image data to numpy array and standardize values (divide by 255 since RGB values ranges from 0 to 255)
data_lowres = np.array(data_lowres, dtype="float") / 255.0

# Show data shape
print("Shape of data_lowres: ", data_lowres.shape)

# Display 10 real images
fig, axs = plt.subplots(2, 5, sharey=False, tight_layout=True, figsize=(16,9), facecolor='white')
n=0
for i in range(0,2):
    for j in range(0,5):
        axs[i,j].matshow(data_lowres[n])
        n=n+1
plt.show()

scaler=MinMaxScaler(feature_range=(-1, 1))

# Select images that we want to use for model trainng
data=data_lowres.copy()
print("Original shape of the data: ", data.shape)

data=data.reshape(-1, 1)
scaler.fit(data)
data=scaler.transform(data)
data=data.reshape(data_lowres.shape[0], 64, 64, 3)
print("Shape of the scaled array: ", data.shape)


def generator(latent_dim):
    model = Sequential(name="Generator")

    # Hidden Layer 1: Start with 8 x 8 image
    n_nodes = 8 * 8 * 128
    model.add(Dense(n_nodes, input_dim=latent_dim, name='Generator-Hidden-Layer-1'))
    model.add(Reshape((8, 8, 128), name='Generator-Hidden-Layer-Reshape-1'))

    # Hidden Layer 2: Upsample to 16 x 16
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                              name='Generator-Hidden-Layer-2'))
    model.add(ReLU(name='Generator-Hidden-Layer-Activation-2'))

    # Hidden Layer 3: Upsample to 32 x 32
    model.add(Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                              name='Generator-Hidden-Layer-3'))
    model.add(ReLU(name='Generator-Hidden-Layer-Activation-3'))

    # Hidden Layer 4: Upsample to 64 x 64
    model.add(Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same',
                              name='Generator-Hidden-Layer-4'))
    model.add(ReLU(name='Generator-Hidden-Layer-Activation-4'))

    model.add(Conv2D(filters=3, kernel_size=(5, 5), activation='tanh', padding='same', name='Generator-Output-Layer'))
    return model


latent_dim = 100
gen_model = generator(latent_dim)

# Show model summary and plot model diagram
gen_model.summary()
plot_model(gen_model, show_shapes=True, show_layer_names=True, dpi=400)


def discriminator(in_shape=(64, 64, 3)):
    model = Sequential(name="Discriminator")

    # Hidden Layer 1
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape,
                     name='Discriminator-Hidden-Layer-1'))
    model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-1'))

    # Hidden Layer 2
    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape,
                     name='Discriminator-Hidden-Layer-2'))
    model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-2'))

    # Hidden Layer 3
    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape,
                     name='Discriminator-Hidden-Layer-3'))
    model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-3'))

    # Hidden Layer 4
    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape,
                     name='Discriminator-Hidden-Layer-4'))
    model.add(LeakyReLU(alpha=0.2, name='Discriminator-Hidden-Layer-Activation-4'))

    # Flatten and Output Layers
    model.add(Flatten(name='Discriminator-Flatten-Layer')) 
    model.add(Dropout(0.3,
                      name='Discriminator-Flatten-Layer-Dropout'))
    model.add(Dense(1, activation='sigmoid', name='Discriminator-Output-Layer'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model

dis_model = discriminator()

dis_model.summary()
plot_model(dis_model, show_shapes=True, show_layer_names=True, dpi=400)


def def_gan(generator, discriminator):
    discriminator.trainable = False

    model = Sequential(name="DCGAN")
    model.add(generator)
    model.add(discriminator)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model

gan_model = def_gan(gen_model, dis_model)

gan_model.summary()
plot_model(gan_model, show_shapes=True, show_layer_names=True, dpi=400)


def real_samples(n, dataset):
    # Samples of real data
    X = dataset[np.random.choice(dataset.shape[0], n, replace=True), :]

    # Class labels
    y = np.ones((n, 1))
    return X, y


def latent_vector(latent_dim, n):
    latent_input = np.random.randn(latent_dim * n)

    # Reshape into a batch of inputs for the network
    latent_input = latent_input.reshape(n, latent_dim)
    return latent_input


def fake_samples(generator, latent_dim, n):
    latent_output = latent_vector(latent_dim, n)

    X = generator.predict(latent_output)

    y = np.zeros((n, 1))
    return X, y


def performance_summary(generator, discriminator, dataset, latent_dim, i, n=50):
    x_real, y_real = real_samples(n, dataset)
    _, real_accuracy = discriminator.evaluate(x_real, y_real, verbose=0)
    x_fake, y_fake = fake_samples(generator, latent_dim, n)
    _, fake_accuracy = discriminator.evaluate(x_fake, y_fake, verbose=0)

    print("*** Evaluation ***")
    print("Discriminator Accuracy on REAL images: ", real_accuracy)
    print("Discriminator Accuracy on FAKE (generated) images: ", fake_accuracy)

    x_fake_inv_trans = x_fake.reshape(-1, 1)
    x_fake_inv_trans = scaler.inverse_transform(x_fake_inv_trans)
    x_fake_inv_trans = x_fake_inv_trans.reshape(n, 64, 64, 3)

    plt.imshow(x_fake_inv_trans[0])
    plt.savefig('image_at_epoch_{:04d}.png'.format(i))
    plt.show()
    fig, axs = plt.subplots(2, 3, sharey=False, tight_layout=True, figsize=(12, 6), facecolor='white')
    k = 0
    for i in range(0, 2):
        for j in range(0, 3):
            axs[i, j].matshow(x_fake_inv_trans[k])
            k = k + 1
    plt.show()


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=2001, n_batch=32, n_eval=100):
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):

        x_real, y_real = real_samples(half_batch, dataset)
        x_fake, y_fake = fake_samples(g_model, latent_dim, half_batch)

        X, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
        discriminator_loss, _ = d_model.train_on_batch(X, y)

        x_gan = latent_vector(latent_dim, n_batch)

        y_gan = np.ones((n_batch, 1))

        generator_loss = gan_model.train_on_batch(x_gan, y_gan)

        if (i) % n_eval == 0:
            print("Epoch number: ", i)
            print("*** Training ***")
            print("Discriminator Loss ", discriminator_loss)
            print("Generator Loss: ", generator_loss)
            performance_summary(g_model, d_model, dataset, latent_dim,i)

train(gen_model, dis_model, gan_model, data, latent_dim)

import imageio.v2 as imageio
import glob
# Generate and save the GIF
anim_file = 'generated_gif.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)





