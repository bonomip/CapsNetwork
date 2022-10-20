from matplotlib import pyplot as plt
import tensorflow as tf

def show_image(x, y, size):
    plt.figure()
    plt.xlabel(y)
    plt.imshow(tf.reshape(x, [1, size, size, 1])[0])