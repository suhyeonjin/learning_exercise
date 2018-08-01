import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


class preprocess:
    def __init__(self):
        pass
        

    def Mnist2data(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        x_data, y_data, x_test, y_test = self.mnist.train.images, self.mnist.train.labels, self.mnist.test.images, self.mnist.test.labels
        print ('num: ',self.mnist.train.num_examples)
        return x_data, y_data, x_test, y_test
        

if __name__ == '__main__':
    pp = preprocess()
    print(pp.Mnist2data())      




'''
def csv_to_dataset(model = "cnn", csvpath = "./bitcoin_ticker.csv", fromidx = 39000) :

    def df_norm(df) :
        newdf = (df - df.mean()) /(df.max() - df.min())
        return newdf - newdf.min()

    data = pd.read_csv('./bitcoin_ticker.csv')
    data = data[data['market'] == 'korbit']
    data = data[data['rpt_key'] == 'btc_krw']
    data = data[['last', 'volume']]

    data_norm = df_norm(data)
    data_pretty = data_norm[fromidx:] # 2017년 6월 28일 데이터부터 사용

    data_pretty_values= data_pretty.values

    #plt.plot(data_pretty)
    #plt.show()

    x_data = []
    y_data = []

    if model=="cnn" :
        for i in range(30, data_pretty.shape[0]-6) :
            x_data.append( df_norm( data_pretty[i-30: i] ).values )  #30 min

            p1 = data_pretty_values[i,0]
            p2 = data_pretty_values[i+5, 0]

            result = int(p2 > p1)   # y=1 => 5분뒤 오른다  / y=0 => 5분뒤 내린다
            y_data.append(result)

        x_data = np.array(x_data, np.float32)
        y_data = np.array(y_data, np.float32)
        y_data = np.reshape(y_data, [data_pretty.shape[0]-36, 1])

    elif model=="rnn" :
        for i in range(30, data_pretty.shape[0]-6) :
            x_data.append( data_pretty[i-30: i].values )  #30 min

            p1 = data_pretty_values[i,0]
            p2 = data_pretty_values[i+5,0]

            result = int(p2 > p1)   # y=1 => 5분뒤 오른다  / y=0 => 5분뒤 내린다
            y_data.append(result)

        x_data = np.array(x_data, np.float32)
        y_data = np.array(y_data, np.float32)
        y_data = np.reshape(y_data, [data_pretty.shape[0]-36, 1])

    return x_data, y_data
'''