import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras import Model
from skimage.transform import resize


def get_inceptionv3_embedding(images):
    """get the embedding from a pretrained inception v3 model"""
    inceptionv3 = InceptionV3()
    embedding = Model(inputs=inceptionv3.input,
                      output=inceptionv3.layers[-2].output)
    embedding_list = list()
    for i, image in enumerate(images):
        # resize with nearest neighbor interpolation
        new_image = resize(image, (299, 299, 3), 0)
        new_image = new_image.reshape(1, 299, 299, 3)
        # embed
        x_embedding = embedding.predict(new_image)
        # store
        embedding_list.append(x_embedding)
    return np.asarray(embedding_list).reshape(images.shape[0], 2048)
