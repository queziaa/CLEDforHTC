import os
import tensorflow as tf
import numpy as np
from .utils import get_shape

# 尝试从tensorflow.contrib.rnn导入LSTMStateTuple，如果失败，则从tf.nn.rnn_cell导入
try:
  from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
  LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple

# 定义双向RNN函数
def bidirectional_rnn(cell_fw, cell_bw, inputs, input_lengths,
                      initial_state_fw=None, initial_state_bw=None,
                      scope=None):
  # 创建一个变量作用域
  with tf.variable_scope(scope or 'bi_rnn') as scope:
    # 使用tf.nn.bidirectional_dynamic_rnn创建一个双向RNN
    # 它会返回前向和后向的输出以及状态
    (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=cell_fw,
      cell_bw=cell_bw,
      inputs=inputs,
      sequence_length=input_lengths,
      initial_state_fw=initial_state_fw,
      initial_state_bw=initial_state_bw,
      dtype=tf.float32,
      scope=scope
    )
    # 将前向和后向的输出沿着第二个维度（axis=2）拼接起来
    outputs = tf.concat((fw_outputs, bw_outputs), axis=2)

    # 定义一个函数，用于拼接前向和后向的状态
    def concatenate_state(fw_state, bw_state):
      # 如果状态是LSTMStateTuple类型，那么分别拼接其c和h属性
      if isinstance(fw_state, LSTMStateTuple):
        state_c = tf.concat(
          (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
        state_h = tf.concat(
          (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
        state = LSTMStateTuple(c=state_c, h=state_h)
        return state
      # 如果状态是tf.Tensor类型，那么直接拼接
      elif isinstance(fw_state, tf.Tensor):
        state = tf.concat((fw_state, bw_state), 1,
                          name='bidirectional_concat')
        return state
      # 如果状态是元组类型，且前向和后向状态的长度相同，那么逐个拼接它们的元素
      elif (isinstance(fw_state, tuple) and
            isinstance(bw_state, tuple) and
            len(fw_state) == len(bw_state)):
        # multilayer
        state = tuple(concatenate_state(fw, bw)
                      for fw, bw in zip(fw_state, bw_state))
        return state
      # 如果状态的类型未知，那么抛出一个错误
      else:
        raise ValueError(
          'unknown state type: {}'.format((fw_state, bw_state)))

    # 拼接前向和后向的状态
    state = concatenate_state(fw_state, bw_state)
    # 返回输出和状态
    return outputs, state

def masking(scores, sequence_lengths, score_mask_value=tf.constant(-np.inf)):
  # 创建一个掩码，形状为 [batch_size, sequence_length]
  score_mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(scores)[1])
  # 创建一个掩码值张量，形状和scores相同
  score_mask_values = score_mask_value * tf.ones_like(scores)
  # 使用掩码选择scores或score_mask_values中的元素
  return tf.where(score_mask, scores, score_mask_values)

def partial_softmax(logits, weights, dim, name,):
  with tf.variable_scope('partial_softmax'):
    # 计算logits的指数
    exp_logits = tf.exp(logits)
    # 如果logits和weights的形状相同，则直接相乘
    # 否则，先扩展weights的维度，然后再相乘
    if len(exp_logits.get_shape()) == len(weights.get_shape()):
      exp_logits_weighted = tf.multiply(exp_logits, weights)
    else:
      exp_logits_weighted = tf.multiply(exp_logits, tf.expand_dims(weights, -1))
    # 计算加权logits的和
    exp_logits_sum = tf.reduce_sum(exp_logits_weighted, axis=dim, keep_dims=True)
    # 计算部分softmax得分
    partial_softmax_score = tf.div(exp_logits_weighted, exp_logits_sum, name=name)
    return partial_softmax_score

def attention(inputs, att_dim, sequence_lengths, scope=None):
  assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

  with tf.variable_scope(scope or 'attention'):
    # 创建一个注意力权重变量，形状为 [att_dim, 1]
    word_att_W = tf.get_variable(name='att_W', shape=[att_dim, 1])

    # 对输入进行投影，形状为 [batch_size, sequence_length, att_dim]
    projection = tf.layers.dense(inputs, att_dim, tf.nn.tanh, name='projection')

    # 计算alpha，形状为 [batch_size, sequence_length]
    alpha = tf.matmul(tf.reshape(projection, shape=[-1, att_dim]), word_att_W)
    alpha = tf.reshape(alpha, shape=[-1, get_shape(inputs)[1]])
    alpha = masking(alpha, sequence_lengths, tf.constant(-1e15, dtype=tf.float32))
    alpha = tf.nn.softmax(alpha)

    # 计算输出，形状为 [batch_size, embedding_dim]
    outputs = tf.reduce_sum(inputs * tf.expand_dims(alpha, 2), axis=1)
    return outputs, alpha

def attention_pre(inputs, sequence_lengths, pre_att, scope=None):
  assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

  with tf.variable_scope(scope or 'attention'):
    # 创建一个注意力权重变量，形状为 [batch_size, sequence_length, embedding_dim]
    word_att_W = tf.tile(pre_att, [1, get_shape(inputs)[1]])
    word_att_W = tf.reshape(word_att_W, [-1, get_shape(inputs)[1], get_shape(pre_att)[1]])

    # 计算alpha，形状为 [batch_size, sequence_length]
    alpha = tf.multiply(inputs, word_att_W)
    alpha = tf.reduce_sum(alpha,-1)
    alpha = masking(alpha, sequence_lengths, tf.constant(-1e15, dtype=tf.float32))
    alpha = tf.nn.softmax(alpha)

    # 计算输出，形状为 [batch_size, embedding_dim]
    outputs = tf.reduce_sum(inputs * tf.expand_dims(alpha, 2), axis=1)
    return outputs, alpha

def attention_class(inputs, class_embedding, input_mask, sequence_lengths, class_num, class_embedding_dim, scope=None):
  # 确保输入的形状是3D的，且最后一维有值
  assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

  with tf.variable_scope(scope or 'attention_class'):
    # 扩展输入掩码的维度，形状变为 [batch_size, sequence_length, 1]
    x_mask = tf.expand_dims(input_mask, axis=-1)
    # 输入张量，形状为 [batch_size, sequence_length, embedding_dim]
    x_emb_0 = inputs
    # 将输入张量和掩码相乘，形状为 [batch_size, sequence_length, embedding_dim]
    x_emb_1 = tf.multiply(x_emb_0, x_mask)

    # 对输入张量进行L2标准化，形状为 [batch_size, sequence_length, embedding_dim]
    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2)
    # 转置类别嵌入，形状为 [embedding_dim, class_num]
    W_class_tran = tf.transpose(class_embedding, [1, 0])
    # 对转置后的类别嵌入进行L2标准化，形状为 [embedding_dim, class_num]
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim=0)
    # 计算输入张量和类别嵌入的点积，形状为 [batch_size, sequence_length, class_num]
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm)

    # 定义卷积核的形状
    filter_shape = [3, class_num, class_num]
    # 创建卷积核权重变量W，形状为 [3, class_num, class_num]
    W = tf.get_variable(name='W', shape=filter_shape,
                        initializer=tf.truncated_normal_initializer(stddev=0.1),
                        dtype=tf.float32, trainable=True)
    # 创建偏置变量b，形状为 [class_num]
    b = tf.get_variable(name='b', shape=[class_num],
                        initializer=tf.constant_initializer(0.1),
                        dtype=tf.float32, trainable=True)

    # 执行卷积操作，形状为 [batch_size, sequence_length, class_num]
    conv = tf.nn.conv1d(G, W, stride=1, padding='SAME', name='conv')
    # 添加偏置，形状为 [batch_size, sequence_length, class_num]
    Att_v_1 = tf.nn.bias_add(conv, b)
    # 对最后一维求最大值，形状为 [batch_size, sequence_length]
    Att_v = tf.reduce_max(Att_v_1, axis=-1)

    # 对注意力值进行掩码操作，形状为 [batch_size, sequence_length]
    Att_v_max = masking(Att_v, sequence_lengths, tf.constant(-1e15, dtype=tf.float32))
    # 对注意力值进行softmax操作，形状为 [batch_size, sequence_length]
    Att_v_max = tf.nn.softmax(Att_v_max)

    # 将注意力值与输入掩码相乘，形状为 [batch_size, sequence_length, 1]
    Att_v_max = tf.multiply(tf.expand_dims(Att_v_max, -1), x_mask)
    # 将输入张量与注意力值相乘，形状为 [batch_size, sequence_length, embedding_dim]
    x_att = tf.multiply(x_emb_0, Att_v_max)
    # 对最后一维求和，形状为 [batch_size, embedding_dim]
    outputs = tf.reduce_sum(x_att, axis=1)
    # 返回输出和注意力值
    return outputs, Att_v_max

# 这个函数attention_class是一个用于计算类别注意力的函数，它的输入包括输入向量、类别嵌入、输入掩码、序列长度、类别数量和类别嵌入维度。函数的主要步骤如下：

# 首先，函数通过扩展输入掩码和输入向量的元素级乘法来应用输入掩码。
# 然后，函数对结果进行L2标准化，并对类别嵌入进行转置和L2标准化。
# 接着，函数计算输入向量和类别嵌入的点积，得到一个新的张量G。
# 函数创建一个卷积层，其权重和偏置由可训练的变量W和b表示。
# 使用G作为输入，通过卷积层进行卷积操作，然后添加偏置，得到Att_v_1。
# 对Att_v_1沿着最后一个维度取最大值，得到Att_v。
# 使用masking函数对Att_v进行掩码操作，然后应用softmax函数，得到Att_v_max。
# 将Att_v_max与输入掩码进行元素级乘法，然后与原始输入向量进行元素级乘法，得到x_att。
# 最后，沿着第一个维度对x_att进行求和，得到输出向量。