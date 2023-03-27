import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import mnist
import json, codecs

(x_train, _), (x_test, _) = fashion_mnist.load_data()

#Normalization

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#To use keras

x_train = x_train.reshape((len(x_train), x_train.shape[1:][0]*x_train.shape[1:][1]))
x_test = x_test.reshape((len(x_test), x_test.shape[1:][0]*x_test.shape[1:][1]))

input_img = Input(shape = (784,))

encoder = Dense(32, activation="relu")(input_img)

encoder = Dense(16, activation="relu")(encoded)

decoder = Dense(32, activation="relu")(encoded)

decoder = Dense(784, activation="sigmoid")(decoded)

autoencoder = Model(input_img,decoded)

autoencoder.compile(optimizer="rmsprop",loss="binary_crossentropy")

hist = autoencoder.fit(x_train,
                       x_train,
                       epochs=200,
                       batch_size=256,
                       shuffle=True,
                       validation_data = (x_train,x_train))
autoencoder.save_weights("autoencoder_model.h5")

with open("autoencoders_hist.json","w") as f:
    json.dump(hist.history,f)

with codecs.open("autoencoders_hist.json","r", encoding="utf-8")  as f:
    n = json.loads(f.read())

encoder = Model(input_img,encoded)
encoded_img = encoder.predict(x_test)
decoded_imgs = autoender.predict(x_test)

