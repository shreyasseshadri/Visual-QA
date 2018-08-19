import tensorflow as tf
import numpy as np

corpus=["This is a sentence","The king is royal","she is the royal queen"]
words=[]
sentences=[]
for sent in corpus:
    sentences.append(sent.split(" "))
    for i in sent.split(" "):
        if i!="." or i!="\n" or i!="-":
            words.append(i)
words=set(words)
word2int = {}
int2word = {}
vocab_size = len(words) # gives the total number of unique words
for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word
# print(sentences)
data = []
WINDOW_SIZE = 3
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
            if nb_word != word:
                data.append([word, nb_word])

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

x_train = [] 
y_train = [] 

for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))

X_train = np.asarray(x_train)
y_train = np.asarray(y_train)
print(X_train.shape)

#Tf model
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 5 # Same as number of features
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) 
hidden_representation = tf.add(tf.matmul(x,W1), b1)
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

sess = tf.Session()

# sess.run(init) 
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy_loss)
init = tf.global_variables_initializer()

n_iters = 2
with tf.Session() as sess:
    sess.run(init)    
    for _ in range(n_iters):
        sess.run(train_step, feed_dict={x: X_train, y_label: y_train})
        print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
    vectors = sess.run(W1 + b1)
    print(vectors)

def get_vector(word):
    assert type(word)==type("h")
    return vectors[word2int[word]]

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word, vectors):
    min_dist = 10000 
    min_index = -1
    query_vector = get_vector(word)
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index
# print(int2word[find_closest('queen',vectors)])