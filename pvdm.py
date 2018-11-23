import io
import itertools
import random
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

MISSING_TOKEN = 'MISSING_TOKEN'


# Class for generating batches for pv-dm model training
class PVDMGenerator(object):
    def __init__(self, word2id, id2par, window=6, batch_size=128):
        self.word2id = word2id
        self.id2par = id2par
        self.window = window
        self.batch_size = batch_size

    def get_word_id(self, w):
        return self.word2id.get(w, self.word2id[MISSING_TOKEN])

    def next_sample(self):
        par_size = len(self.id2par)
        while(True):
            parid = random.randint(0, par_size-1)
            par = self.id2par[parid]
            i = random.randint(0, len(par) - self.window)
            win = [ self.get_word_id(w) for w in par[i:i+self.window-1] ]
            yield (parid, win, self.get_word_id(par[i+self.window-1]))

    def next_batch(self):
        Xpar = []
        Xwin = []
        y = []
        for sample in self.next_sample():
            Xpar.append(sample[0])
            Xwin.append(sample[1])
            y.append(sample[2])
            if len(y) == self.batch_size:
                yield (np.array(Xpar).reshape((self.batch_size, 1)),
                       np.array(Xwin).reshape((self.batch_size, self.window-1)),
                       np.array(y).reshape((self.batch_size, 1)))
                Xpar = []
                Xwin = []
                y = []


class PVDM(object):
    def __init__(self, word_vectors_file, word_vocab_size=100000, par_vector_size=300):
        self.word_vocab_size = word_vocab_size
        self.par_vector_size = par_vector_size

        self.load_word_vectors(word_vectors_file)

        word_tokenizer = CountVectorizer().build_tokenizer()
        self.tokenize = lambda s: word_tokenizer(s.lower())

        self.dense_weights = None
        self.dense_bias = None

    # construct tf graph and do gradient descent
    def _train(self, docs, window=6, batch_size=128, lr=0.01, tol=1e-3, max_iter=1000, freeze=False):
        self.build_paragraph_vocab(docs, window)
        pvdm_generator = PVDMGenerator(self.word2id, self.id2par, window, batch_size)

        tf.reset_default_graph()

        Xpar = tf.placeholder(tf.int64, shape=(batch_size, 1))
        Xwin = tf.placeholder(tf.int64, shape=(batch_size, window-1))
        y = tf.placeholder(tf.int64, shape=(batch_size, 1))

        par_embeddings = tf.get_variable("par_embeddings", shape=(self.par_vocab_size, self.par_vector_size),
                                         initializer=tf.random_normal_initializer())
        word_embeddings = tf.get_variable("word_embeddings", shape=self.word_embeddings.shape,
                                      initializer=tf.constant_initializer(self.word_embeddings),
                                      trainable=not freeze)

        par_vectors = tf.reshape(tf.nn.embedding_lookup(par_embeddings, Xpar), [batch_size, -1])
        win_vectors = tf.reshape(tf.nn.embedding_lookup(word_embeddings, Xwin), [batch_size, -1])

        concat = tf.concat([par_vectors, win_vectors], axis=1)
        if self.dense_weights is not None and self.dense_bias is not None:
            logits = tf.layers.dense(concat, self.word_vocab_size, name="dense", trainable=not freeze,
                                     kernel_initializer=tf.constant_initializer(self.dense_weights),
                                     bias_initializer=tf.constant_initializer(self.dense_bias))
        else:
            logits = tf.layers.dense(concat, self.word_vocab_size, name="dense", trainable=not freeze)
        with tf.variable_scope("dense", reuse=True):
            weights = tf.get_variable("kernel")
            bias = tf.get_variable("bias")
        loss = tf.losses.sparse_softmax_cross_entropy(y, logits)

        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            prev_loss_value = np.inf
            for i in range(max_iter):
                batch = next(pvdm_generator.next_batch())
                _, loss_value = sess.run((train, loss), feed_dict={Xpar: batch[0], Xwin: batch[1], y: batch[2]})
                if i % 10 == 0 and abs(prev_loss_value - loss_value) < tol:
                    break
                if i % 100 == 0:
                    print("iter: %d, loss: %f" % (i, loss_value))
                prev_loss_value = loss_value
            print("iter: %d, loss: %f" % (i, loss_value))

            # save for reuse during inference
            self.word_embeddings, self.par_embeddings = sess.run((word_embeddings, par_embeddings))
            self.dense_weights, self.dense_bias = sess.run((weights, bias))

    def train(self, docs, window=6, batch_size=128, lr=0.01, tol=1e-3, max_iter=1000):
        self._train(docs, window=window, batch_size=batch_size, lr=lr, tol=tol, max_iter=max_iter)
        return self.par_embeddings

    def vectorize(self, docs, window=6, batch_size=128, lr=0.01, tol=1e-3, max_iter=1000):
        self._train(docs, window=window, batch_size=batch_size, lr=lr, tol=tol, max_iter=max_iter, freeze=True)
        return self.par_embeddings

    def normalize_paragraph(self, p):
        return p.strip().lower()

    def build_paragraph_vocab(self, docs, window):
        docs = [ self.normalize_paragraph(d) for d in docs ]
        self.id2par = {}
        cnt = 0
        for d in docs:
            dt = self.tokenize(d)
            if len(dt) < window:
                dt.extend([MISSING_TOKEN] * (window-len(dt)))
            self.id2par[cnt] = dt
            cnt += 1
        self.par_vocab_size = cnt

    def load_word_vectors(self, word_vectors_file):
        fin = io.open(word_vectors_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        embeddings = []
        self.word2id = {}
        self.word_vector_size = d
        cnt = 0
        for line in fin:
            tokens = line.rstrip().split(' ')
            self.word2id[tokens[0]] = cnt
            embeddings.append(list(map(float, tokens[1:])))
            cnt += 1
            if cnt >= self.word_vocab_size-1:
                break
        embeddings.append([0] * self.word_vector_size)
        self.word2id[MISSING_TOKEN] = cnt
        self.word_embeddings = np.array(embeddings).reshape((self.word_vocab_size, self.word_vector_size))
