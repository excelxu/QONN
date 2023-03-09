import os
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

# 来自远古的传承
# 古代的量子计算曾用经典计算机计算梯度进行优化
def ideal_vector_generate(label):
    vector = {
        1:[1.0,0.],
        2:[0.,1.]
    }

    return vector.get(label)

# RBS(+)的求导
def rot_gate_bp_grad(local_input_state, local_bp_error, theta):
    '''return grad on theta point'''
    Th = theta
    grad = local_bp_error[0] * (-np.sin(Th) * local_input_state[0] + np.cos(Th) * local_input_state[1]) \
           + local_bp_error[1] * (-np.cos(Th) * local_input_state[0] - np.sin(Th) * local_input_state[1])

    return grad

def Relu(x):
    return np.maximum(x, 0)
def R_Relu(input):
    # R_Relu函数误差逆传播矩阵
    #   此处显示详细说明
    output = input
    length = len(input)
    for i in range(length):
        if input[i] > 1e-20:
            output[i] = 1
        else:
            output[i] = 1e-20

    return [output]

def updater(theta, state_input, zeta_0, zeta_1, zeta_2, zeta_3, zeta_4, label, Rate):
    '''根据输入的θ，ζ(sets),和数据的标签，输出优化后的角度'''
    # θ的重新命名
    theta_1 = theta[0]
    theta_2 = theta[1]
    theta_3 = theta[2]
    theta_4 = theta[3]
    theta_5 = theta[4]
    # zetas都是四维列表
    layer_input = zeta_0
    layer_output = zeta_4
    # 根据标签生成优化目标矢量（Ideal_V）
    ideal_vector = ideal_vector_generate(label)
    #   zeta_list = np.column_stack((zeta_0,zeta_1,zeta_2,zeta_3,zeta_4))
    # 进行激活函数处理，得到被激活的输出
    activated_output = Relu(layer_output)
    result = activated_output[2:4]
    # 计算LOSS
    error_vector = [0., 0., (result[0] - ideal_vector[0]) ** 2, (result[1] - ideal_vector[1]) ** 2]
    loss = sum(error_vector)
    re_relu = R_Relu(zeta_4)
    re_relu = re_relu[0]
    # 计算每一层的error_vector
    delta_4 = [0.,
               0.,
               re_relu[2] * 2 * (result[0] - ideal_vector[0]),
               re_relu[3] * 2 * (result[1] - ideal_vector[1])
               ]
    delta_3 = [delta_4[0],
               delta_4[1] * np.cos(theta_5) + delta_4[2] * np.sin(-theta_5),
               delta_4[1] * np.sin(theta_5) + delta_4[2] * np.cos(theta_5),
               delta_4[3]
               ]
    delta_2 = [delta_3[0] * np.cos(theta_3) + delta_3[1] * np.sin(-theta_3),
               delta_3[0] * np.sin(theta_3) + delta_3[1] * np.cos(theta_3),
               delta_3[2] * np.cos(theta_4) + delta_3[3] * np.sin(-theta_4),
               delta_3[2] * np.sin(theta_4) + delta_3[3] * np.cos(theta_4),
               ]
    delta_1 = [delta_2[0],
               delta_2[1] * np.cos(theta_2) + delta_2[2] * np.sin(-theta_2),
               delta_2[1] * np.sin(+theta_2) + delta_2[2] * np.cos(theta_2),
               delta_2[3]
               ]
    # delta_0 =
    # 计算BP
    delta = [delta_1, delta_2, delta_3, delta_4]
    zeta_mid = [0, 0, 0, 0]
    zeta_mid[0] = zeta_2[0]
    zeta_mid[1] = zeta_2[1]
    zeta_mid[2] = zeta_3[2]
    zeta_mid[3] = zeta_3[3]

    theta_1_grad = rot_gate_bp_grad(state_input[0:2], delta_1[0:2], theta_1)  # theta_1
    theta_2_grad = rot_gate_bp_grad(zeta_0[1:3], delta_2[1:3], theta_2)  # theta_2
    theta_3_grad = rot_gate_bp_grad(zeta_1[0:2], delta_3[0:2], theta_3)  # theta_3
    theta_4_grad = rot_gate_bp_grad(zeta_1[2:4], delta_3[2:4], theta_4)  # theta_4
    theta_5_grad = rot_gate_bp_grad(zeta_mid[1:3], delta_4[1:3], theta_5)  # theta_5

    # 输出的新theta
    theta_1_updated = theta_1 - Rate * theta_1_grad  # theta_1
    theta_2_updated = theta_2 - Rate * theta_2_grad  # theta_2
    theta_3_updated = theta_3 - Rate * theta_3_grad  # theta_3
    theta_4_updated = theta_4 - Rate * theta_4_grad  # theta_4
    theta_5_updated = theta_5 - Rate * theta_5_grad  # theta_5

    theta_grad = np.array([theta_1_grad,
                  theta_2_grad,
                  theta_3_grad,
                  theta_4_grad,
                  theta_5_grad
                  ])
    theta_updated = [theta_1_updated,
                     theta_2_updated,
                     theta_3_updated,
                     theta_4_updated,
                     theta_5_updated,
                     loss
                     ]

    return (delta, theta_grad, theta_updated, loss)
def Standard(x):
    """Z-score normaliaztion"""
    x = (x - np.mean(x)) / np.std(x)
    return x

dev = qml.device('default.qubit', wires=[0,1,2,3,'aux']) # 定义量子模拟器,包含自动微分模式

def RBS(on_wire, theta):
    if type(on_wire) is tuple:
        qml.Hadamard(wires=on_wire[0])
        qml.Hadamard(wires=on_wire[1])
        qml.CZ(wires=on_wire)
        qml.RY(theta / 2, wires=on_wire[0])
        qml.RY(-theta / 2, wires=on_wire[1])
        qml.CZ(wires=on_wire)
        qml.Hadamard(wires=on_wire[0])
        qml.Hadamard(wires=on_wire[1])
    else:
        raise('RBS作用线路编号为元组形式')

def data_to_state(data):
    # 设定计算基的矢量模式
    basis10000 = np.zeros(2**5)
    basis01000 = np.zeros(2**5)
    basis00100 = np.zeros(2**5)
    basis00010 = np.zeros(2**5)
    basis00001 = np.zeros(2**5)
    basis10000[2**4] = 1.0
    basis01000[2**3] = 1.0
    basis00100[2**2] = 1.0
    basis00010[2**1] = 1.0
    basis00001[2**0] = 1.0
    state =  data[0]*basis10000\
            +data[1]*basis01000\
            +data[2]*basis00100\
            +data[3]*basis00010\
            +data[4]*basis00001 # 第五位是辅助量子比特上的振幅，一般为0.2
    return state

def qubit_state_to_data(state):
    # updater 中应用的都是四维矢量
    data = [0, 0, 0, 0]
    data[0] = np.real(state[16])
    data[1] = np.real(state[8])
    data[2] = np.real(state[4])
    data[3] = np.real(state[2])
    # data[0] = data[0].numpy()
    # data[1] = data[1].numpy()
    # data[2] = data[2].numpy()
    # data[3] = data[3].numpy()
    return data

'''
# 为了更好地使用qml,初始化量子态必须
# (1)通过生成电路生成
# (2)作为全局变量导入
'''

@qml.qnode(dev)
def layer_42_state(ini_state, params):
    # 装载初始数据
    # 拥有五条线路，最下面一条是辅助量子比特
    qml.QubitStateVector(ini_state
                        ,wires=[0,1,2,3,'aux'])
    # 金字塔电路主体
    RBS((0, 1), params[0])
    qml.Snapshot("zeta_0")
    RBS((1, 2), params[1])
    qml.Snapshot("zeta_1")
    RBS((0, 1), params[2])
    qml.Snapshot("zeta_2")
    RBS((2, 3), params[3])
    qml.Snapshot("zeta_3")
    RBS((1, 2), params[4])
    return qml.state()

# 测试集测试函数
def test_acc(test_data, param):
    count = 0.0
    for i in range(80):
        data = test_data.values[i, (0, 1, 2, 3)]
        data = Standard(data)
        feat = np.sqrt(0.8) * (data) / np.sqrt(np.dot(data, data))
        f = [feat[0], feat[1], feat[2], feat[3], np.sqrt(0.2)]
        data = f  # 输入态
        label = test_data.values[i, 4]  # 标签

    # ---量子电路的输出
        ini_state = data_to_state(data)
        out_state = layer_42_state(ini_state, param)
        out_data = qubit_state_to_data(out_state)
        out3 = Relu(out_data[2])
        out4 = Relu(out_data[3])
        if label == 1 and out3>out4:
            count = count+1.0
        if label == 2 and out3<out4:
             count = count+1.0
    acc = count/80.0
    return acc
'''
测试🐔，1！5！
'''

if __name__ == '__main__':
    k=0
    # 读取excel中数据
    # ab_train = pd.read_excel('IrisData/ab20_train_100.xlsx') # pd默认将第一行作为标题
    # ab_test = pd.read_excel('IrisData/irisAB_test_80.xlsx')
    bc_test = pd.read_excel('IrisData/irisBC_test_80.xlsx')
    bc_train = pd.read_excel('IrisData/bc20_train_100.xlsx')
    # 量子网络的初始化
    # ini_state = data_to_state([0, 0.8, 0.6, 0, 0])
    param = np.array([1.39,2.70,1.22,1.55,1.42]) # 初始化参数
    lr = 0.25# 学习率
    sum_grad = np.zeros(5)
    acc_on_test = np.zeros(11)
    loss_training = np.zeros(11)
    sum_loss = 0

    for i in range(100): # 遍历一次训练集,ab（100个）
        # 每次10次对梯度做一次升级
        data = bc_train.values[i,(0,1,2,3)]
        data = Standard(data)
        feat = np.sqrt(0.8)*(data)/np.sqrt(np.dot(data,data))
        f = [feat[0], feat[1], feat[2], feat[3], np.sqrt(0.2)]
        data = f # 输入态
        label = bc_train.values[i,4] #标签
        #---量子电路的输出与梯度
        ini_state = data_to_state(data)
        # 网络运行
        out = qml.snapshots(layer_42_state)(ini_state, param)  # 同时记录最终结果和中间的数据，使用字典存储
        zeta_state_0 = out.get('zeta_0')
        zeta_state_1 = out.get('zeta_1')
        zeta_state_2 = out.get('zeta_2')
        zeta_state_3 = out.get('zeta_3')
        zeta_state_4 = out.get('execution_results')
        #zeta_5 = out.get()

        state_input = [feat[0], feat[1], feat[2], feat[3]]
        zeta_0 = qubit_state_to_data(zeta_state_0)
        zeta_1 = qubit_state_to_data(zeta_state_1)
        zeta_2 = qubit_state_to_data(zeta_state_2)
        zeta_3 = qubit_state_to_data(zeta_state_3)
        zeta_4 = qubit_state_to_data(zeta_state_4)

        # 计算梯度与loss
        (delta, grad, theta_updated, loss) = updater(param, state_input, zeta_0, zeta_1, zeta_2, zeta_3, zeta_4, label, lr)
        sum_loss = sum_loss + loss
        sum_grad = sum_grad + grad
        if i == 0:
            acc_on_test[0] = test_acc(bc_test, param)
            # loss_training[0] = sum_loss
        if (i+1)%10 == 0:  # 每10个样本升级一次
            k = k + 1
            param = param - lr*sum_grad
            # loss_training[k] = sum_loss
            acc = test_acc(bc_test, param)
            acc_on_test[k] = acc
            sum_grad = 0
            sum_loss = 0
    plt.plot(acc_on_test)
    plt.show()
    # plt.plot(loss_training)
    # plt.show()

        # 根据标签设置目标矢量,1:[1,0];2:[0,1]
        # print(theta_grad)


