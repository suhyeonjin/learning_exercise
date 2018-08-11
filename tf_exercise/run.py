import tensorflow as tf
import numpy as np
import sys

from model import models
from data import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

#name, dataset_path, learning_rate, batch_size, epoch
def run_model(name, learning_rate, batch_size, epoch):
    
    #make model
    sess = tf.Session()
    md = models.Model(sess, name, FLAGS)

    #data preprocessing
    dp = utils.preprocess(FLAGS)
    x_data, y_data, x_test, y_test = dp.Mnist2data()
    
    #initialize
    sess.run(tf.global_variables_initializer())

    #training
    for ep in range(epoch):
        avg_cost = 0
        total_batches = int(x_data.shape[0]/batch_size)
        #print(int(x_data.shape[0]), total_batches, batch_size)

        for i in range(total_batches):
            x,y = x_data[i*batch_size:i*batch_size+batch_size], y_data[i*batch_size:i*batch_size+batch_size]
            cost, _ = md.train(x,y)
            avg_cost += cost/batch_size

        print('[*] epoch:', '%04d' % (ep + 1), ', cost =', '{:.9f}'.format(avg_cost))
    print("[*] learning Finishied!")

    #testset predict
    print('Accuracy:', md.get_accuracy(x_test, y_test))
    return True


if __name__ == "__main__":
    #Setting global variable
<<<<<<< HEAD
    #flags.DEFINE_string("name", "unnamed", "model name for ckpt file")

=======
    flags.DEFINE_string("dataset_path", "./data", "dataset directory")
    
>>>>>>> d08849c9be3c09e1b4e74e2c756eacab03639612
    #hyper parameter
    flags.DEFINE_integer("batch_size", 100, "batch_size")
    flags.DEFINE_integer("epoch",15,"total_epoches")
    flags.DEFINE_float("learning_rate",0.001, "learning_rate")

    #define model
    flags.DEFINE_string("model",'cnn',"Select Model")

    if FLAGS.model == 'cnn':
        run_model("M1", FLAGS.learning_rate, FLAGS.batch_size, FLAGS.epoch)
        
    else:
        sys.exit(0)
