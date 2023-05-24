import tensorflow as tf
import numpy as np
import math
import pdb

""" ======== Non Layers ========= """

def initializer(init, shape):# 这是初始化一下变量
    if init == "zero":
        return tf.zeros(shape, dtype=tf.float32)# 填充一个形状是shape的数据类型是float32的数据
    elif init == "he":

        fan_in = np.prod(shape[0:-1])# 数据乘法
        std = 1 / np.sqrt(fan_in)
        return tf.random.uniform(shape, minval=-std, maxval=std, dtype=tf.float32)
# tf.random.uniform：从均匀分布中输出随机值。生成的值在该[minval, maxval)范围内遵循均匀分布



class GCN_NODE_WEIGHT(tf.keras.Model):# 图卷积节点权重
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.Wc = tf.Variable(initializer("he", (in_dims, out_dims)), trainable=True) # (in_dims, out_dims)
        self.Wn = tf.Variable(initializer("he", (in_dims, out_dims)), trainable=True) # (in_dims, out_dims)
        self.We = tf.Variable(initializer("he", (2, out_dims)), trainable=True) # (2, out_dims)
        self.q = tf.Variable(initializer("he", (out_dims, 1)), trainable=True)
        self.b = tf.Variable(initializer("zero", (out_dims,)), trainable=True)
# trainable：可训练的

# tf.Variable:
# 为了区分需要计算梯度信息的张量与不需要计算梯度信息的张量，
# TensorFlow 增加了一种专门的数据类型来支持梯度信息的记录：tf.Variable。
# tf.Variable 类型在普通的张量类型基础上添加了 name，trainable 等属性来支持计算图的构建。
# 由于梯度运算会消耗大量的计算资源，而且会自动更新相关参数
# 对于不需要的优化的张量，如神经网络的输入X，不需要通过 tf.Variable 封装；
# 相反，对于需要计算梯度并优化的张量，如神经网络层的W和b，
# 需要通过 tf.Variable 包裹以便 TensorFlow 跟踪相关梯度信息

    def call(self, x, adj, edge, training):
        Zc = tf.matmul(x, self.Wc)
        v_Wn = tf.matmul(x, self.Wn)
        e_We = tf.tensordot(edge, self.We, axes=[[2], [0]]) # (n_verts, n_nbors, filters)

        nh_sizes = tf.expand_dims(tf.math.count_nonzero(adj + 1, axis=1, dtype=tf.float32), -1)
        neighbor = tf.gather(v_Wn, adj) + e_We
        # neighbor = tf.gather(v_Wn, adj)
        weight = tf.nn.softmax(tf.matmul(neighbor, self.q))

        # Zn = tf.divide(tf.reduce_sum(neighbor * weight, 1),
        #             tf.maximum(nh_sizes, tf.ones_like(nh_sizes))) # (in_dims, out_dims)
        Zn = tf.reduce_sum(neighbor * weight, 1) # 一定方式计算张量中元素之和
        
        h = tf.nn.relu(Zn + Zc+ self.b)# 使用relu激活函数
        if training:
            h = tf.nn.dropout(h, 0.4)
            # 每个神经元被丢弃的概率是0.4
            # 在这里并不是真正被丢掉，而是在这一轮的训练中不更新这个神经元的权值，权值在这一轮训练被保留，下一轮训练可能又会被更新

        return h

# tf.keras.Model:
# tf.keras.Model类将定义好的网络结构封装入一个对象，用于训练、测试和预测



class Dense(tf.keras.Model): # FC层在keras中叫做Dense层

# 通过继承Model类进行实例化，
# 需要定义自己的__init__并且在call方法中实现网络的前向传播结构
# (即在这个方法中定义网络结构)。

    def __init__(self, in_dims, out_dims, is_relu):
        super().__init__()
        self.W = tf.Variable(initializer("he", [in_dims, out_dims]), trainable=True)
        self.b = tf.Variable(initializer("zero", [out_dims]), trainable=True)
        self.is_relu = is_relu

    def call(self, x, training):
        if training:
            x = tf.nn.dropout(x, 0.4)
        Z = tf.matmul(x, self.W) + self.b
        if self.is_relu:
            Z = tf.nn.relu(Z)

        return Z



class Sinkhorn(tf.keras.Model):#最优传输，用于W距离
    def __init__(self, d=256, max_iter=10, epsilon=1e-12, lamda=1.0):
        super().__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.lamda = lamda


    def cos(self, x1, x2):
        M = tf.matmul(x1, tf.transpose(x2, [0, 2, 1]))
        a = tf.sqrt(tf.reduce_sum(tf.math.square(x1), axis=-1, keepdims=True))
        b = tf.sqrt(tf.reduce_sum(tf.math.square(x2), axis=-1, keepdims=True))

        norm = tf.matmul(a, tf.transpose(b, [0, 2, 1]))
        dist = M / (norm + self.epsilon)
        return dist


    def call(self, x1, x2):
        M = 1 - self.cos(x1, x2)
        K = tf.exp(-M * self.lamda)


        batch_size = x1.get_shape()[0]
        len1 = x1.get_shape()[1]
        len2 = x2.get_shape()[1]
        w1 = tf.ones((batch_size, len1), dtype=tf.float32) / len1
        w2 = tf.ones((batch_size, len2), dtype=tf.float32) / len2
        ui = tf.ones((batch_size, len1), dtype=tf.float32)
        vi = tf.ones((batch_size, len2), dtype=tf.float32)
        ui = tf.expand_dims(ui, axis=-1) 
        vi = tf.expand_dims(vi, axis=-1)
        w1 = tf.expand_dims(w1, axis=-1)
        w2 = tf.expand_dims(w2, axis=-1)

        for i in range(self.max_iter):
            vi = tf.math.divide(w2, tf.matmul(tf.transpose(K, [0, 2, 1]), ui) + self.epsilon)
            ui = tf.math.divide(w1, tf.matmul(K, vi) + self.epsilon)

        G = ui * K * tf.transpose(vi, [0, 2, 1])
        
        G_max = tf.reduce_max(G, axis=[-1, -2], keepdims=True)
        G_min = tf.reduce_min(G, axis=[-1, -2], keepdims=True)
        # pdb.set_trace()
        G = tf.math.divide(G - G_min, G_max - G_min + self.epsilon)
        
        return G * M

   
class MultiHeadAttention(tf.keras.Model):# 多头注意力
    def __init__(self, hid_dim, n_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        self.fc_q = tf.keras.layers.Dense(hid_dim * n_heads)
        self.fc_k = tf.keras.layers.Dense(hid_dim * n_heads)
        self.fc_v = tf.keras.layers.Dense(hid_dim * n_heads)
        self.fc_o = tf.keras.layers.Dense(hid_dim)
        self.scale = tf.sqrt(tf.cast(hid_dim, dtype=tf.float32))

        self.sinkhorn = Sinkhorn()


    def call(self, query, key, value):
        n1 = query.shape[0]
        n2 = value.shape[0]
        Q = self.fc_q(query) # (n1, hid_dim)
        K = self.fc_q(key) # (n2, hid_dim)
        V = self.fc_q(value) # (n2, hid_dim)

        Q = tf.reshape(Q, [n1, self.n_heads, self.hid_dim]) # (n1, n_heads, hid_dim)
        K = tf.reshape(K, [n2, self.n_heads, self.hid_dim]) # (n2, n_heads, hid_dim)
        V = tf.reshape(V, [n2, self.n_heads, self.hid_dim]) # (n2, n_heads, hid_dim)

        # s = tf.matmul(tf.transpose(Q, [1, 0, 2]), tf.transpose(K, [1, 2, 0])) / self.scale # (n_heads, n1, n2)
        s = self.sinkhorn(tf.transpose(Q, [1, 0, 2]), tf.transpose(K, [1, 0, 2])) # (n_heads, n1, n2)

        attention = tf.nn.softmax(s * 10, axis=-1) # (n_heads, n1, n2)


        x = tf.matmul(attention, tf.transpose(V, [1, 0, 2])) # (n_heads, n1, hid_dim)
        x = tf.reshape(tf.transpose(x, [1, 0, 2]), [n1, -1]) # (n1, n_heads*hid_dim)
        x = self.fc_o(x)# (n1, hid_dim)

        return s, x


class PW_classifier(tf.keras.Model):
    def __init__(self, in_dims, gcn_layer_num, gcn_config):
        super().__init__()
        self.gcn_layer_num = gcn_layer_num
        self.sinkhorn = Sinkhorn()


        self.gcn1 = (GCN_NODE_WEIGHT(in_dims, 256), GCN_NODE_WEIGHT(in_dims, 256))
        self.gcn2 = (GCN_NODE_WEIGHT(256, 512), GCN_NODE_WEIGHT(256, 512))
        # self.gcn3 = (GCN_NODE_WEIGHT(512, 512), GCN_NODE_WEIGHT(512, 512))
        # self.gcn4 = (GCN_NODE_WEIGHT(512, 512), GCN_NODE_WEIGHT(512, 512))

        self.gcn_cross = Dense(1024, 512, True)
        self.gcn_final = (GCN_NODE_WEIGHT(512, 512), GCN_NODE_WEIGHT(512, 512))

        self.dense1 = Dense(1024, 512, True)
        self.dense2 = Dense(512, 1, False)

        self.transformer = MultiHeadAttention(512)


    @tf.function
    def call(self, x0, adj0, e0, x1, adj1, e1, examples, training):
        x0 = tf.cast(x0, dtype=tf.float32)
        x1 = tf.cast(x1, dtype=tf.float32)
        e0 = tf.cast(e0, dtype=tf.float32)
        e1 = tf.cast(e1, dtype=tf.float32)
        
        x0 = self.gcn1[0](x0, adj0, e0, training)
        x1 = self.gcn1[1](x1, adj1, e1, training)
        x0 = self.gcn2[0](x0, adj0, e0, training)
        x1 = self.gcn2[1](x1, adj1, e1, training)
        # x0 = self.gcn3[0](x0, adj0, e0, training)
        # x1 = self.gcn3[1](x1, adj1, e1, training)
        # x0 = self.gcn4[0](x0, adj0, e0, training)
        # x1 = self.gcn4[1](x1, adj1, e1, training)


        for i in range(2):
            s1, x0_encoding = self.transformer(x0, x1, x1)
            s2, x1_encoding = self.transformer(x1, x0, x0)
            x0_new = self.gcn_cross(tf.concat((x0, x0_encoding), axis=-1), training)
            x1_new = self.gcn_cross(tf.concat((x1, x1_encoding), axis=-1), training)
            x0 = x0_new
            x1 = x1_new
        
            x0 = self.gcn_final[0](x0, adj0, e0, training)
            x1 = self.gcn_final[1](x1, adj1, e1, training)


        # merge layer
        out1 = tf.gather(x0, examples[:, 0])
        out2 = tf.gather(x1, examples[:, 1])
        output1 = tf.concat([out1, out2], axis=0)
        output2 = tf.concat([out2, out1], axis=0)
        output = tf.concat((output1, output2), axis=1)

        # dense layer
        output = self.dense1(output, training)
        out = self.dense2(output, training)

        # # average layer
        out = tf.reduce_mean(tf.stack(tf.split(out, 2)), 0)
        out = tf.nn.sigmoid(out)

        return (out, (s1, s2))


class Weight_Cross_Entropy(tf.keras.Model):
    def __init__(self, pn_ratio=0.5):
        super().__init__()
        self.pn_ratio = pn_ratio
        self.cerition = tf.keras.losses.BinaryCrossentropy()

    def call(self, preds, labels):
        labels = tf.cast(labels, dtype=tf.float32)
        preds, sim = preds
        scale_vector = (self.pn_ratio * (labels - 1) / -2) + ((labels + 1) / 2)

        labels = (labels + 1) / 2
        labels = tf.expand_dims(labels, -1)
        loss = tf.keras.losses.binary_crossentropy(labels, preds)
        loss = tf.reduce_mean(loss * scale_vector)

        return loss

     
