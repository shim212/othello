# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.functions import *


class OthelloNet:
    """
    네트워크 구성은 아래와 같음
        [input layers]
        Conv1 (64 filters, 3x3)- BatNor1 - Relu1 -
        Conv2 (64 filters, 3x3)- BatNor2 - Relu2 -      
        Conv3 (64 filters, 3x3)- BatNor3 - Relu3 -
        Conv4 (64 filters, 3x3)- BatNor4 - Relu4 -
        Conv5 (64 filters, 3x3)- BatNor5 - Relu5 -

        Conv6 (64 filters, 3x3)- BatNor6 - Relu6 -
        Conv7 (64 filters, 3x3)- BatNor7 - Relu7 -      
        Conv8 (64 filters, 3x3)- BatNor8 - Relu8 -
        Conv9 (64 filters, 3x3)- BatNor9 - Relu9 -
        Conv10 (64 filters, 3x3)- BatNor10 - Relu10 -

        [policy head]
        Conv11 (2 filters, 1x1) - BatNor11 - Relu11 -
        Affine13 (64 = 8x8)
        cf. loss: soft max - cross entropy error

        [value head]
        Conv12 (1 filter, 1x1) - BatNor12 - Relu12-
        Affine14 (128) - Relu14 -
        Affine15 (1) - Tanh
        cf. loss: mean squared error
    """

    def __init__(self, input_dim=(1, 2, 8, 8),
                 conv_param_1 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},

                 conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_7 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_8 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_9 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_10 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 
                 conv_param_11 = {'filter_num':2, 'filter_size':1, 'pad':0, 'stride':1},
                 conv_param_12 = {'filter_num':1, 'filter_size':1, 'pad':0, 'stride':1},
                 Affine13_size = 64, Affine14_size = 128, Affine15_size = 1):
        
        # 가중치 초기화===========
        # 각 층의 뉴런 하나당 앞 층의 몇 개 뉴런과 연결되는가（TODO: 자동 계산되게 바꿀 것）
        pre_node_nums = np.array([2*3*3, 64*3*3, 64*3*3, 64*3*3, 64*3*3, 64*3*3, 64*3*3, 64*3*3, 64*3*3, 64*3*3, 64*1*1, 64*1*1, 2*8*8, 1*8*8, 128])
        weight_init_scales = np.sqrt(1.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값(He) => Xavier로 바꾸기
        
        self.params = {}
        pre_channel_num = input_dim[1]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6, conv_param_7, conv_param_8, conv_param_9, conv_param_10, conv_param_11]):
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size']).astype(np.float16)
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num']).astype(np.float16)
            pre_channel_num = conv_param['filter_num']

        self.params['W12'] = weight_init_scales[11] * np.random.randn(conv_param_12['filter_num'], 64, conv_param_12['filter_size'], conv_param_12['filter_size']).astype(np.float16)
        self.params['b12'] = np.zeros(conv_param_12['filter_num']).astype(np.float16)

        self.params['W13'] = weight_init_scales[12] * np.random.randn(2*8*8, Affine13_size).astype(np.float16)
        self.params['b13'] = np.zeros(Affine13_size).astype(np.float16)
        self.params['W14'] = weight_init_scales[13] * np.random.randn(1*8*8, Affine14_size).astype(np.float16)
        self.params['b14'] = np.zeros(Affine14_size).astype(np.float16)
        self.params['W15'] = weight_init_scales[14] * np.random.randn(Affine14_size, Affine15_size).astype(np.float16)
        self.params['b15'] = np.zeros(Affine15_size).astype(np.float16)

        for i in range(12):
            self.params['BN' + str(i+1) + '_RM'] = None
            self.params['BN' + str(i+1) + '_RV'] = None

        
        # 계층 생성===========

        # [Input Layers]
        self.inputlayers = OrderedDict()
        self.inputlayers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param_1['stride'], conv_param_1['pad'])
        self.inputlayers['BatNor1'] = BatchNormalization()
        self.inputlayers['Relu1'] = Relu()

        self.inputlayers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param_2['stride'], conv_param_2['pad'])
        self.inputlayers['BatNor2'] = BatchNormalization()
        self.inputlayers['Relu2'] = Relu()

        self.inputlayers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], conv_param_3['stride'], conv_param_3['pad'])
        self.inputlayers['BatNor3'] = BatchNormalization()
        self.inputlayers['Relu3'] = Relu()

        self.inputlayers['Conv4'] = Convolution(self.params['W4'], self.params['b4'], conv_param_4['stride'], conv_param_4['pad'])
        self.inputlayers['BatNor4'] = BatchNormalization()
        self.inputlayers['Relu4'] = Relu()

        self.inputlayers['Conv5'] = Convolution(self.params['W5'], self.params['b5'], conv_param_5['stride'], conv_param_5['pad'])
        self.inputlayers['BatNor5'] = BatchNormalization()
        self.inputlayers['Relu5'] = Relu()

        self.inputlayers['Conv6'] = Convolution(self.params['W6'], self.params['b6'], conv_param_6['stride'], conv_param_6['pad'])
        self.inputlayers['BatNor6'] = BatchNormalization()
        self.inputlayers['Relu6'] = Relu()

        self.inputlayers['Conv7'] = Convolution(self.params['W7'], self.params['b7'], conv_param_7['stride'], conv_param_7['pad'])
        self.inputlayers['BatNor7'] = BatchNormalization()
        self.inputlayers['Relu7'] = Relu()

        self.inputlayers['Conv8'] = Convolution(self.params['W8'], self.params['b8'], conv_param_8['stride'], conv_param_8['pad'])
        self.inputlayers['BatNor8'] = BatchNormalization()
        self.inputlayers['Relu8'] = Relu()

        self.inputlayers['Conv9'] = Convolution(self.params['W9'], self.params['b9'], conv_param_9['stride'], conv_param_9['pad'])
        self.inputlayers['BatNor9'] = BatchNormalization()
        self.inputlayers['Relu9'] = Relu()

        self.inputlayers['Conv10'] = Convolution(self.params['W10'], self.params['b10'], conv_param_10['stride'], conv_param_10['pad'])
        self.inputlayers['BatNor10'] = BatchNormalization()
        self.inputlayers['Relu10'] = Relu()


        # [Policy Head]
        self.policyhead = OrderedDict()
        self.policyhead['Conv11'] = Convolution(self.params['W11'], self.params['b11'], conv_param_11['stride'], conv_param_11['pad'])
        self.policyhead['BatNor11'] = BatchNormalization()
        self.policyhead['Relu11'] = Relu()
        self.policyhead['Affine13'] = Affine(self.params['W13'], self.params['b13'])

        self.policyhead_lastlayer = SoftmaxWithLoss()
        
        
        # [Value Head]
        self.valuehead = OrderedDict()
        self.valuehead['Conv12'] = Convolution(self.params['W12'], self.params['b12'], conv_param_12['stride'], conv_param_12['pad'])
        self.valuehead['BatNor12'] = BatchNormalization()
        self.valuehead['Relu12'] = Relu()
        self.valuehead['Affine14'] = Affine(self.params['W14'], self.params['b14'])
        self.valuehead['Relu14'] = Relu()
        self.valuehead['Affine15'] = Affine(self.params['W15'], self.params['b15'])
        self.valuehead['Tanh'] = Tanh()        

        self.valuehead_lastlayer = MeanSquaredError()


    def predict(self, x, train_flg=False):
        for layer in self.inputlayers.values():
            x = layer.forward(x, train_flg)

        p = x
        for layer in self.policyhead.values():
            p = layer.forward(p, train_flg)

        v = x
        for layer in self.valuehead.values():
            v = layer.forward(v, train_flg)

        return p, v



    def loss_policy(self, x, t, train_flg=False):
        p, v = self.predict(x, train_flg)
        loss_policy = self.policyhead_lastlayer.forward(p, t)
        return loss_policy

    def loss_value(self, x, t, train_flg=False):
        p, v = self.predict(x, train_flg)
        loss_value = self.valuehead_lastlayer.forward(v, t)
        return loss_value

    def loss_total(self, x, t_policy, t_value, train_flg=False):
        p, v = self.predict(x, train_flg)
        loss_policy = self.policyhead_lastlayer.forward(p, t_policy)
        loss_value = self.valuehead_lastlayer.forward(v, t_value)
        return loss_policy + loss_value


    def accuracy_policy(self, x, t):
        p, v = self.predict(x, train_flg=False)
        p = np.argmax(p, axis = 1)
        t = np.argmax(t, axis=1)
        acc_policy = np.sum(p == t) / t.shape[0]
        return acc_policy


    def gradient_policy(self, x, t):
        # forward
        self.loss_policy(x, t, train_flg=True)

        # backward

        p_layers = list(self.policyhead.values())
        p_layers.reverse()

        i_layers = list(self.inputlayers.values())
        i_layers.reverse()


        # [Policy]

        dout = 1
        dout = self.policyhead_lastlayer.backward(dout)
        
        for layer in p_layers:
            dout = layer.backward(dout)

        for layer in i_layers:
            dout = layer.backward(dout)

        grads = {}
        for i in range(1, 11):
            grads['W' + str(i)] = self.inputlayers['Conv'+str(i)].dW
            grads['b' + str(i)] = self.inputlayers['Conv'+str(i)].db

        grads['W11'] = self.policyhead['Conv11'].dW
        grads['b11'] = self.policyhead['Conv11'].db

        grads['W13'] = self.policyhead['Affine13'].dW
        grads['b13'] = self.policyhead['Affine13'].db

        return grads


    def gradient_value(self, x, t):
        # forward
        self.loss_value(x, t, train_flg=True)


        # backward

        v_layers = list(self.valuehead.values())
        v_layers.reverse()

        i_layers = list(self.inputlayers.values())
        i_layers.reverse()


        # [Value]

        dout = 1
        dout = self.valuehead_lastlayer.backward(dout)
        
        for layer in v_layers:
            dout = layer.backward(dout)

        for layer in i_layers:
            dout = layer.backward(dout)

        grads = {}
        for i in range(1, 11):
            grads['W' + str(i)] = self.inputlayers['Conv'+str(i)].dW
            grads['b' + str(i)] = self.inputlayers['Conv'+str(i)].db

        grads['W12'] = self.valuehead['Conv12'].dW
        grads['b12'] = self.valuehead['Conv12'].db

        grads['W14'] = self.valuehead['Affine14'].dW
        grads['b14'] = self.valuehead['Affine14'].db

        grads['W15'] = self.valuehead['Affine15'].dW
        grads['b15'] = self.valuehead['Affine15'].db

        return grads


    def gradient(self, x, t_policy, t_value):
        # forward
        self.loss_total(x, t_policy, t_value, train_flg=True)

        # backward

        p_layers = list(self.policyhead.values())
        p_layers.reverse()

        v_layers = list(self.valuehead.values())
        v_layers.reverse()

        i_layers = list(self.inputlayers.values())
        i_layers.reverse()


        # [Policy]

        dout = 1
        dout = self.policyhead_lastlayer.backward(dout)
        
        for layer in p_layers:
            dout = layer.backward(dout)

        dout_policy = dout
        
        # [Value]

        dout = 1
        dout = self.valuehead_lastlayer.backward(dout)
        
        for layer in v_layers:
            dout = layer.backward(dout)
            
        dout_value = dout

        # [Input]

        dout = (dout_policy + dout_value) / 2

        for layer in i_layers:
            dout = layer.backward(dout)

        grads = {}
        for i in range(1, 11):
            grads['W' + str(i)] = self.inputlayers['Conv'+str(i)].dW
            grads['b' + str(i)] = self.inputlayers['Conv'+str(i)].db

        grads['W11'] = self.policyhead['Conv11'].dW
        grads['b11'] = self.policyhead['Conv11'].db

        grads['W13'] = self.policyhead['Affine13'].dW
        grads['b13'] = self.policyhead['Affine13'].db

        grads['W12'] = self.valuehead['Conv12'].dW
        grads['b12'] = self.valuehead['Conv12'].db

        grads['W14'] = self.valuehead['Affine14'].dW
        grads['b14'] = self.valuehead['Affine14'].db

        grads['W15'] = self.valuehead['Affine15'].dW
        grads['b15'] = self.valuehead['Affine15'].db

        return grads



    def save_params(self, file_name="params.pkl"):
        
        for i in range(10):
            self.params['BN' + str(i+1) + '_RM'] = self.inputlayers['BatNor' + str(i+1)].running_mean
            self.params['BN' + str(i+1) + '_RV'] = self.inputlayers['BatNor' + str(i+1)].running_var

        self.params['BN11_RM'] = self.policyhead['BatNor11'].running_mean
        self.params['BN11_RV'] = self.policyhead['BatNor11'].running_var

        self.params['BN12_RM'] = self.valuehead['BatNor12'].running_mean
        self.params['BN12_RV'] = self.valuehead['BatNor12'].running_var
        
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)



    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i in range(1, 11):
            self.inputlayers['Conv'+str(i)].W = self.params['W' + str(i)]
            self.inputlayers['Conv'+str(i)].b = self.params['b' + str(i)]


        self.policyhead['Conv11'].W = self.params['W11']
        self.policyhead['Conv11'].b = self.params['b11']

        self.policyhead['Affine13'].W = self.params['W13']
        self.policyhead['Affine13'].b = self.params['b13']

        
        self.valuehead['Conv12'].W = self.params['W12']
        self.valuehead['Conv12'].b = self.params['b12']

        self.valuehead['Affine14'].W = self.params['W14']
        self.valuehead['Affine14'].b = self.params['b14']

        self.valuehead['Affine15'].W = self.params['W15']
        self.valuehead['Affine15'].b = self.params['b15']


        for i in range(10):
            self.inputlayers['BatNor' + str(i+1)].running_mean = self.params['BN' + str(i+1) + '_RM']
            self.inputlayers['BatNor' + str(i+1)].running_var = self.params['BN' + str(i+1) + '_RV']

        self.policyhead['BatNor11'].running_mean = self.params['BN11_RM']
        self.policyhead['BatNor11'].running_var = self.params['BN11_RV']

        self.valuehead['BatNor12'].running_mean = self.params['BN12_RM']
        self.valuehead['BatNor12'].running_var = self.params['BN12_RV']
        

