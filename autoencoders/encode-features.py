import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

size = 2

(x_train_val, y_train_val), (x_test, _) = cifar10.load_data()

x_train_val = x_train_val / 255.
x_test = x_test / 255.

x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, random_state=314159)

input_img = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

x = Conv2D(8 * size, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(8 * size, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8 * size, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(16 * size, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16 * size, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16 * size, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32 * size, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32 * size, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32 * size, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
encoded = MaxPooling2D((2, 2))(x)
x = Conv2D(32 * size, (3, 3), activation='relu', padding='same')(encoded)
x = Conv2D(32 * size, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32 * size, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16 * size, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16 * size, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16 * size, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8 * size, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8 * size, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8 * size, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=64,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=[earlystopping])

encoded_imgs = encoder.predict(x_train_val)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0],
                                    encoded_imgs.shape[1] *
                                    encoded_imgs.shape[2] *
                                    encoded_imgs.shape[3])

np.save('encode-features.npy', encoded_imgs)
