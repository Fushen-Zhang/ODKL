import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tensorflow.python import pywrap_tensorflow
from Scripts.RNN_loop import dynamic_rnn
from Scripts.MultiRNNCell import MultiRNNCell
from Scripts.gradient_descent import GradientDescentOptimizer as opt
from Scripts.Neuron_models import SNUELayer,GPLayer
import matplotlib.pyplot as plt

FLOAT_TYPE = tf.float32
T = 24

parser = argparse.ArgumentParser(description = '') 
parser.add_argument('--length_scale', type = int, default = 100, help = 'hyper-parameters of GP: size of basis vector set') 
parser.add_argument('--W', type = float, default = 1/18, help = 'hyper-parameters of GP: horizontal length scale') 
parser.add_argument('--B', type = np.array, default = np.array([1.0]), help = 'hyper-parameters of GP: vertical variation') 
parser.add_argument('--epsilon', type = float, default = 0.15, help = 'hyper-parameters of sparse GP: a pruning threshold') 
parser.add_argument('--sigma', type = float, default = 0.16, help = 'hyper-parameters of GP: human-injected noise') 
parser.add_argument('--lag', type = int, default = 12, help = 'length of lagged time-series features') 
parser.add_argument('--N', default = [12+33,64,64,64], help = 'hyper-parameters of deep sSNN: specify the deep architecture here')  
parser.add_argument('--conf_SNUs', default={'recurrent': False, 'activation': tf.tanh, 'g': tf.identity, 'decay': 0.0}, help='hyper-parameters of deep sSNN: specific arguments for the SNUs')  
parser.add_argument('--lr', type = int, default = 0.002, help = 'learning rate') 
args = parser.parse_args() 

def Create_deep_SNU():
    rnn_layers_OSTL = []
    for i in range(1,len(args.N)):
        initW = np.random.uniform(-np.sqrt(6) / np.sqrt(args.N[i-1] + args.N[i]), np.sqrt(6) / np.sqrt(args.N[i-1] + args.N[i]), size=(args.N[i-1], args.N[i]))
        initH = np.random.uniform(-np.sqrt(6) / np.sqrt(args.N[i] + args.N[i]), np.sqrt(6) / np.sqrt(args.N[i] + args.N[i]), size=(args.N[i], args.N[i]))
        layer = SNUELayer(args.N[i], args.N[i-1], **args.conf_SNUs, initW=initW, initH=initH, name='OSTL_RNN' + str(i))
        rnn_layers_OSTL.append(layer)
    return rnn_layers_OSTL
    
def loss_function(target_t, output_t,time, last_time): 
    loss = tf.losses.mean_squared_error(target_t,output_t)
    return tf.reduce_sum(loss)

def gradient_function(loss, output_t, time, last_time, layer):
    grads = tf.gradients(loss, output_t)[0] 
    return grads

def train_test_split(dataset,hour_feature,window:int):

    train_data = dataset[:240*48]
    test_data =  dataset[240*48:300*48]
    
    scaler = StandardScaler()
    hour_feature = scaler.fit_transform(hour_feature)
    train_hour_feature = hour_feature[:240*48]
    test_hour_feature =  hour_feature[240*48:300*48]
    
    mean, std = np.mean(train_data), np.std(train_data)
    train_data, test_data = (train_data-mean)/std, (test_data-mean)/std

    test_data = np.concatenate((train_data[-window:],test_data))
    test_hour_feature = np.concatenate((train_hour_feature[-window:],test_hour_feature))
    
    train_x, train_y,test_x, test_y  = [], [], [], []
    
    for i in range(window, train_data.shape[0]):
        train_x.append(np.hstack((train_data[i-window:i],train_hour_feature[i].ravel())))
        train_y.append(train_data[i])

    for i in range(window, test_data.shape[0]):
        test_x.append(np.hstack((test_data[i-window:i],test_hour_feature[i].ravel())))
        test_y.append(test_data[i])

    train_x, train_y, test_x, test_y = np.array(train_x),np.array(train_y),np.array(test_x),np.array(test_y)
    
    Offset_train = train_x.shape[0] % 48
    train_x = train_x[Offset_train:]
    train_y = train_y[Offset_train:]
    
    return train_x, train_y, test_x, test_y

def Train(train_X, train_Y, test_X, test_Y, model_path):
    train_X,test_X = np.expand_dims(train_X, 0),np.expand_dims(test_X, 0)    
    train_Y,test_Y = np.expand_dims(train_Y, 0).reshape(1,-1,1),np.expand_dims(test_Y, 0).reshape(1,-1,1)
    train_input_X, train_input_Y = np.concatenate((train_X,test_X),axis=1), np.concatenate((train_Y,test_Y),axis=1)
    
    tf.reset_default_graph()
    rnn_layers = Create_deep_SNU()
    gp_layer = GPLayer(length_scale = args.length_scale-1, initW = args.W, initB = args.B, epsilon = args.epsilon, sigma_0 = args.sigma ) 
                    
    rnn_layers.append(gp_layer)
    multi_rnn_cell = MultiRNNCell(rnn_layers)
    
    with tf.name_scope("example"):
        x = tf.placeholder(FLOAT_TYPE, shape=[1,None,args.N[0]]) 
        y = tf.placeholder(FLOAT_TYPE, shape=[1,None,1])
        train = tf.placeholder_with_default(tf.constant(1, tf.int32), ())
        ini_state = (
        tf.placeholder(FLOAT_TYPE, shape=[1, None, args.N[-1]]),
        tf.placeholder(FLOAT_TYPE, shape=[1, None, None]),
        tf.placeholder(FLOAT_TYPE, shape=[1, None, None]),
        tf.placeholder(FLOAT_TYPE, shape=[1, None, None]),
        tf.placeholder(FLOAT_TYPE, shape=[None, 1]),
        tf.placeholder(FLOAT_TYPE, shape=[1]),
        tf.placeholder(FLOAT_TYPE, shape=None),
        tf.placeholder(FLOAT_TYPE, shape=[1, 1, 1]),
        tf.placeholder(FLOAT_TYPE, shape=[1, 1, 1]),
        )
        optimizer = opt(learning_rate = args.lr)
        mu, state = dynamic_rnn(multi_rnn_cell, x, y, loss_function, gradient_function, train, 1, 0, 1, optimizer,initial_state=ini_state, dtype=tf.float32)
        loss = loss_function(y, mu, None, 0)
        train_step_OSTL = tf.no_op()

    with tf.Session() as sess:    
        sess.run(tf.global_variables_initializer())
        reader = pywrap_tensorflow.NewCheckpointReader(model_path)
        trainable_variables = tf.trainable_variables() 
        for v in trainable_variables:
            if 'gp_layer' not in v.name:
                sess.run(tf.assign(v, reader.get_tensor(v.name[:-2])))  
        
        inher_GPstate = (   np.zeros((1,1, args.N[-1]), dtype=np.float32),
                            np.zeros((1,1,1), dtype=np.float32),
                            np.zeros((1,1,1), dtype=np.float32),
                            np.zeros((1,1,1), dtype=np.float32),
                            np.zeros((1,1), dtype=np.float32),
                            np.zeros((1), dtype=np.float32),
                            np.zeros(None, dtype=np.float32),
                            np.zeros((1,1,1), dtype=np.float32),
                            np.zeros((1,1,1), dtype=np.float32),
                        )

        forecasts = []
        for day in range(train_input_X.shape[1] // T):
            trainX_day = train_input_X[:,day*T:(day+1)*T, :]
            trainY_day = train_input_Y[:,day*T:(day+1)*T, :]
            
            fd_train = {x: trainX_day, y: trainY_day,ini_state:inher_GPstate}
            [train_mu, train_state, _, _] = sess.run([mu, state, loss, train_step_OSTL], feed_dict = fd_train)
            inher_GPstate = train_state[-1]
            forecasts.extend(np.array(train_mu).squeeze())  
                    
    return np.array(forecasts).squeeze()[-test_Y.shape[1]:], train_state[-1][4].squeeze()[-test_Y.shape[1]:]
    
if __name__ == "__main__":
    # an example based on Ausgrid Resident dataset
    # task: forecasting small-scale aggregated loads (aggregated loads of 30 households)

    # %% preprocessing
    path = '2012-2013 Solar home electricity data v2.csv' # 
    load_data = pd.read_csv(path)
    load_data = load_data.loc[load_data['Consumption Category']=='GC']
    time_index = load_data.columns[-48:]

    data = []
    for i in range(300):
        user_i = load_data.loc[load_data['Customer']==(i+1)]
        user_i = np.array(user_i[time_index].iloc[:,:])
        if user_i.shape[0]==365:
            data.append(user_i.flatten(order='C'))

    aggre_load = np.zeros_like(data[0])
    for i in range(30):
        aggre_load = aggre_load + data[i]

    user_1 = load_data.loc[load_data['Customer']==1]
    new_date = []
    for i in range(len(user_1['date'])):
        date_new = user_1['date'].iloc[i]
        if '/' in date_new[:2]:
            day = '0' + date_new[:1]
        else:
            day = date_new[:2]
        new_date.append(date_new[-4:]+'/'+date_new[-7:-5]+'/'+day)

    datetime = []
    for i in range(365):
        datetime_i = []
        for j in range(48):
                datetime_i.append(new_date[i] + ' ' + time_index[j])
        datetime.extend(datetime_i)
    datetime = pd.DatetimeIndex(datetime)
        
    hour_onehot = pd.get_dummies(datetime.hour, drop_first=False, prefix="hour")
    min_onehot = pd.get_dummies(datetime.minute, drop_first=False, prefix="min")
    dayofweek_onehot = pd.get_dummies(datetime.dayofweek, drop_first=False, prefix="day_of_week")
    
    hour_feature = np.hstack((dayofweek_onehot.iloc[:,:], hour_onehot.iloc[:,:]))
    hour_feature = np.hstack((hour_feature, min_onehot.iloc[:,:]))

    train_x, train_y, test_x, test_y = train_test_split(aggre_load.ravel(), window=args.lag, hour_feature=hour_feature) 
    
    # %% model training
    model_path = './checkpoints/model_saved' # pretrained deep sSNU checkpoints
    yhat, yvar = Train(train_x, train_y, test_x, test_y, model_path)
        
    # %% plot 95% prediction intervals
    low_bound, high_bound = yhat - 1.96*np.sqrt(yvar), yhat + 1.96*np.sqrt(yvar)
    x = np.arange(240)
    plt.plot(x, yhat[:240], label='prediction')
    plt.plot(x, test_y.ravel()[:240], label='true')
    plt.fill_between(x, low_bound[:240], high_bound[:240], facecolor='blue', alpha=0.3) #0.95置信区间
    plt.legend()
    plt.show()
