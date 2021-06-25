import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() #unpacking mnist dataset directly into train test variables.

#Normalizing X variable data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #Output layer

model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#Highest accuracy provided out of various optimizer loss tests.

model.fit(x_train, y_train, epochs=3)

#Running predictions:
model.save('mnist_set_nn.model')
run = tf.keras.models.load_model('mnist_set_nn.model')
predict = run.predict([x_test])

print(np.argmax(predict[0]))
plt.imshow(x_test[0])
plt.show()