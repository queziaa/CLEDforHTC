import tensorflow as tf
import numpy as np
import pickle

# 获取张量的形状
def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims

# 计算训练变量的总参数数量
def count_parameters(trained_vars):
  total_parameters = 0
  print('=' * 100)
  for variable in trained_vars:
    variable_parameters = 1
    for dim in variable.get_shape():
      variable_parameters *= dim.value
    print('{:70} {:20} params'.format(variable.name, variable_parameters))
    print('-' * 100)
    total_parameters += variable_parameters
  print('=' * 100)
  print("Total trainable parameters: %d" % total_parameters)
  print('=' * 100)

# 读取词汇表
def read_vocab(vocab_file):
  print('Loading vocabulary ...')
  with open(vocab_file, 'rb') as f:
    word_to_index = pickle.load(f)
    print('Vocabulary size = %d' % len(word_to_index))
    return word_to_index

# 批量文档标准化
def batch_doc_normalize(docs):
  sent_lengths = np.array([len(doc) for doc in docs], dtype=np.int32)
  max_sent_length = sent_lengths.max()
  word_lengths = [[len(sent) for sent in doc] for doc in docs]
  max_word_length = max(map(max, word_lengths))

  padded_docs = np.zeros(shape=[len(docs), max_sent_length, max_word_length], dtype=np.int32)  # PADDING 0
  word_lengths = np.zeros(shape=[len(docs), max_sent_length], dtype=np.int32)
  for i, doc in enumerate(docs):
    for j, sent in enumerate(doc):
      word_lengths[i, j] = len(sent)
      for k, word in enumerate(sent):
        padded_docs[i, j, k] = word

  return padded_docs, sent_lengths, max_sent_length, word_lengths, max_word_length

# 加载Glove预训练词嵌入
def load_glove(glove_file, emb_size, vocab):
  # 打印加载Glove预训练词嵌入的消息
  print('Loading Glove pre-trained word embeddings ...')
  # 初始化词嵌入权重字典
  embedding_weights = {}
  # 打开Glove文件
  f = open(glove_file, encoding='utf-8')
  # 遍历Glove文件中的每一行
  for line in f:
    # 分割行中的值
    values = line.split()
    # 获取单词
    word = values[0]
    # 获取单词对应的向量，并将其转换为浮点类型的numpy数组
    vector = np.asarray(values[1:], dtype='float32')
    # 将单词和对应的向量添加到词嵌入权重字典中
    embedding_weights[word] = vector
  # 关闭Glove文件
  f.close()
  # 打印词嵌入权重字典的长度和Glove文件的路径
  print('Total {} word vectors in {}'.format(len(embedding_weights), glove_file))

  # 初始化词嵌入矩阵，其形状为(词汇表长度, 嵌入大小)，并且值在-0.5/emb_size到0.5/emb_size之间均匀分布
  embedding_matrix = np.random.uniform(-0.5, 0.5, (len(vocab), emb_size)) / emb_size

  # 初始化未在词嵌入权重字典中找到的单词数量
  oov_count = 0
  # 遍历词汇表中的每一个单词和对应的索引
  for word, i in vocab.items():
    # 获取单词对应的词嵌入向量
    embedding_vector = embedding_weights.get(word)
    # 如果词嵌入向量存在
    if embedding_vector is not None:
      # 将词嵌入矩阵中对应索引的行设置为词嵌入向量
      embedding_matrix[i] = embedding_vector
    # 如果词嵌入向量不存在
    else:
      # 未在词嵌入权重字典中找到的单词数量加1
      oov_count += 1
  # 打印未在词嵌入权重字典中找到的单词数量
  print('Number of OOV words = %d' % oov_count)

  # 返回词嵌入矩阵
  return embedding_matrix