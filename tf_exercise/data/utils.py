<<<<<<< HEAD
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class preprocess:
    def __init__(self):
        pass
=======
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

class preprocess:
    def __init__(self, FLAGS):
        self.dataset_path = FLAGS.dataset_path        
>>>>>>> d08849c9be3c09e1b4e74e2c756eacab03639612

    def Mnist2data(self):
        self.mnist = input_data.read_data_sets(self.dataset_path+"/MNIST_data/", one_hot=True)
        x_data, y_data, x_test, y_test = self.mnist.train.images, self.mnist.train.labels, self.mnist.test.images, self.mnist.test.labels
        #print ('num: ',self.mnist.train.num_examples)
        return x_data, y_data, x_test, y_test

if __name__ == '__main__':
    pp = preprocess()
<<<<<<< HEAD
    print(pp.Mnist2data())      
=======
    print(pp.Mnist2data())
>>>>>>> d08849c9be3c09e1b4e74e2c756eacab03639612
