import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class preprocess:
    def __init__(self, FLAGS):
        self.dataset_path = FLAGS.dataset_path        

    def Mnist2data(self):
        self.mnist = input_data.read_data_sets(self.dataset_path+"/MNIST_data/", one_hot=True)
        x_data, y_data, x_test, y_test = self.mnist.train.images, self.mnist.train.labels, self.mnist.test.images, self.mnist.test.labels
        return x_data, y_data, x_test, y_test

if __name__ == '__main__':
    pp = preprocess()
    print(pp.Mnist2data())
