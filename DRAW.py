import numpy
import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from glob import glob
import os
import random
import re


# TODO: validation + early stopping. Maybe hold-out + cross validation the rest?

# active memory?
# https://papers.nips.cc/paper/6295-can-active-memory-replace-attention.pdf

# tensorflow and list
# https://datascience.stackexchange.com/questions/15056/how-to-use-lists-in-tensorflow

# problem with the whole program is the framework doesn't support "ragged" tensors well, see
# https://datascience.stackexchange.com/questions/15056/how-to-use-lists-in-tensorflow
# maybe a new set of mathematical notations?
# TODO: need a memory mechanism
class Draw():
    def __init__(self):
        self.DEBUG = True

        self.img_size = 64
        self.num_channels = 3

        self.attention_n = 24
        self.n_hidden = 128
        self.n_z = 50       # latent code length
        self.num_class = 134        # number of label classes
        self.sequence_length = 3
        self.batch_size = 2 if self.DEBUG else 64
        self.portion_as_training_data = 4/5
        self.share_parameters = False

        self.optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)

        # TODO: variable image size: maybe classify by size and do batch matrix multiplication
        # checkout https://stackoverflow.com/questions/38966533/different-image-sizes-in-tensorflow-with-batch-size-1
        # and https://gist.github.com/eerwitt/518b0c9564e500b4b50f
        self.images = tf.placeholder(tf.float32, [None, None, None, self.num_channels])       # variable image size
        # self.images = None
        # self.load_images("./data/train", "*.jpg")

        # Qsampler noise
        self.e = [tf.random_normal([self.n_z], mean=0, stddev=1) for i in range(self.batch_size)]   # [self.batch_size]

        # What kinda structure? A cell IS A CELL, with vectors as input/output
        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # encoder Op
        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # decoder Op

        # self.canvas = [0] * self.sequence_length
        # canvas: list of tensors [batch_size, time, tensor[canvas_size]]
        self.canvas = []
        # self.mu, self.logsigma, self.sigma = [0] * self.sequence_length, [0] * self.sequence_length, [0] * self.sequence_length
        # mu, logsigma, sigma: [(less or equal than) batch_size, self.sequence_length]
        self.mu, self.logsigma, self.sigma = [], [], []


        # x = tf.reshape(self.images, [-1, self.img_size*self.img_size*self.num_colors])
        # x = self.images     # [batch, height, width, channel]
        self.attn_params = []
        # self.generated_images = []
        self.generation_loss = []

        self.sess = tf.Session()

    # given a hidden decoder layer:
    # locate where to put attention filters
    # TODO: maybe variable window aspect ratio?
    def attn_window(self, scope, h_dec, img_shape):
        # with tf.variable_scope(scope, reuse=self.share_parameters):
        parameters = dense(h_dec, self.n_hidden, 5, scope_name=scope)     # [batch, 5]
        # gx_, gy_: center of 2d gaussian on a scale of -1 to 1
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(parameters, 5, axis=1)   # each: [1(batch), 1]

        gx_ = tf.reshape(gx_, [1])
        gy_ = tf.reshape(gy_, [1])
        log_sigma2 = tf.reshape(log_sigma2, [1])
        log_delta = tf.reshape(log_delta, [1])
        log_gamma = tf.reshape(log_gamma, [1])

        # move gx/gy to be a scale of -imgsize to +imgsize (?)
        # dense() doesn't seem to guarantee positiveness, but that's not a problem. See paper
        # "The scaling...is chosen to ensure that the initial patch...roughly covers the whole image."
        gx = tf.cast((img_shape[1]+1)/2, tf.float32) * (gx_ + 1)
        gy = tf.cast((img_shape[0]+1)/2, tf.float32) * (gy_ + 1)

        sigma2 = tf.exp(log_sigma2)
        # stride/delta: how far apart these patches will be
        # typo?
        # delta = (self.img_size - 1) / ((self.attention_n-1) * tf.exp(log_delta))
        delta = tf.cast((tf.maximum(img_shape[0], img_shape[1]) - 1) / (self.attention_n-1), tf.float32) * tf.exp(log_delta)
        # returns [Fx, Fy, gamma]

        gamma = tf.exp(log_gamma)
        self.attn_params.append([gx, gy, delta, sigma2, gamma])

        return self.filterbank(gx, gy, sigma2, delta, img_shape) + (gamma,)

    # Given a center, distance, and spread
    # Construct [attention_n x attention_n] patches of gaussian filters
    # represented by Fx = horizontal gaussian, Fy = vertical gaussian
    def filterbank(self, gx, gy, sigma2, delta, img_size):
        # 1 x N, look like [[0,1,2,3,4]]
        grid_i = tf.cast(tf.range(self.attention_n), tf.float32)
        # centers for the individual patches
        mu_x = gx + (grid_i - (self.attention_n + 1)/2) * delta     # [self.attention_n]
        mu_y = gy + (grid_i - (self.attention_n + 1)/2) * delta
        mu_x = tf.reshape(mu_x, [self.attention_n, 1])          # [self.attention_n, 1]
        mu_y = tf.reshape(mu_y, [self.attention_n, 1])
        # 1 x 1 x imgsize, looks like [[[0,1,2,3,4,...,27]]]
        im_x = tf.reshape(tf.cast(tf.range(img_size[1]), tf.float32), [1, -1])       # [1, 1, img_size[1]]
        im_y = tf.reshape(tf.cast(tf.range(img_size[0]), tf.float32), [1, -1])
        # list of gaussian curves for x and y
        # sigma2 = tf.reshape(sigma2, [-1, 1, 1])     # [1(batch), 1, 1]
        Fx = tf.exp(-tf.square((im_x - mu_x) / (2*sigma2)))     # [self.attention_n, img_size[1]]
        Fy = tf.exp(-tf.square((im_y - mu_y) / (2*sigma2)))
        # normalize so area-under-curve = 1
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 1, keep_dims=True), 1e-8)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 1, keep_dims=True), 1e-8)
        return Fx, Fy       # [self.attention_n, img_size[1 | 0]]


    # the read() operation without attention
    def read_basic(self, x, x_hat, h_dec_prev):
        return tf.concat([x, x_hat], 1)

    def read_attention(self, x, x_hat, h_dec_prev):
        # per image
        Fx, Fy, gamma = self.attn_window("read", h_dec_prev, tf.shape(x))
        # we have the parameters for a patch of gaussian filters. apply them.
        def filter_img(img, Fx, Fy, gamma):
            # original:
            # Fx,Fy = [64,5,32]
            # img = [64, 32*32*3]

            # now:
            # img: height * width * channel

            # img = tf.reshape(img, [-1, self.img_size, self.img_size, self.num_channels])    # batch * height * width * channel
            # img_t = tf.transpose(img, perm=[3,0,1,2])   # channel * batch * height * width
            img_t = tf.transpose(img, perm=[2, 0, 1])   # [channel, height, width]

            # batch_colors_array = tf.reshape(img_t, [self.num_channels * self.batch_size, self.img_size, self.img_size])
            batch_colors_array = img_t
            Fx_array = tf.stack([Fx, Fx, Fx])       # 3 channels
            Fy_array = tf.stack([Fy, Fy, Fy])

            Fxt = tf.transpose(Fx_array, perm=[0, 2, 1])      # transpose per image

            # Apply the gaussian patches:
            # square patch
            # TODO: variable aspect ratio glimpse?
            glimpse = tf.matmul(Fy_array, tf.matmul(batch_colors_array, Fxt))   # tensor mul
            # glimpse = tf.reshape(glimpse, [self.num_channels, self.batch_size, self.attention_n, self.attention_n])
            glimpse = tf.transpose(glimpse, [1, 2, 0])      # [height, width, channel]
            # reshape: iterate through two tensors simultaneously, and fill the elements
            # glimpse = tf.reshape(glimpse, [self.batch_size, self.attention_n * self.attention_n * self.num_channels])
            glimpse = tf.reshape(glimpse, [1, -1])      # [1, height * width * channel]
            # print(self.sess.run(tf.shape(glimpse)))
            # finally scale this glimpse with the gamma parameter
            return glimpse * tf.reshape(gamma, [-1, 1])
        x = filter_img(x, Fx, Fy, gamma)            # [1, self.attention_n * self.attention_n * channels]
        x_hat = filter_img(x_hat, Fx, Fy, gamma)
        # print(self.sess.run(tf.shape(x)))
        # print(self.sess.run(tf.shape(x_hat)))
        return tf.concat([x, x_hat], 1)        # [1, 2 * self.attention_n * self.attention_n * channels]

    # encode an attention patch
    def encode(self, prev_state, image):
        # update the RNN with image
        with tf.variable_scope("encoder",reuse=self.share_parameters):
            # see https://www.quora.com/What-is-the-meaning-of-%E2%80%9CThe-number-of-units-in-the-LSTM-cell
            # and https://stackoverflow.com/questions/36732877/about-lstm-cell-state-size
            # feeds a fixed size attention window
            # print(self.sess.run(tf.shape(image)))
            # print(self.sess.run(tf.shape(prev_state)))
            _, new_state_tuple = self.lstm_enc(image, prev_state)     # tuple of [1(batch), self.n_hidden]
            # hidden_layer, next_state = self.lstm_enc(image, prev_state)     # each: self.n_hidden?

        # map the RNN hidden state to latent variables
        # potential bug? dense() modifies scope if not provided
        # with tf.variable_scope("mu", reuse=self.share_parameters):
        mu = dense(new_state_tuple.c, self.n_hidden, self.n_z, scope_name="mu")   # [self.n_z]
        # with tf.variable_scope("sigma", reuse=self.share_parameters):
        logsigma = dense(new_state_tuple.c, self.n_hidden, self.n_z, scope_name="sigma")
        with tf.variable_scope("sigma", reuse=self.share_parameters):
            sigma = tf.exp(logsigma)
        return mu, logsigma, sigma, new_state_tuple


    def sampleQ(self, mu, sigma, i):
        return mu + sigma*self.e[i]        # element-wise multiplication of two vectors (dim=1)

    def decode_layer(self, prev_state, latent):
        # update decoder RNN with latent var
        with tf.variable_scope("decoder", reuse=self.share_parameters):
            _, new_state_tuple = self.lstm_dec(latent, prev_state)

        return new_state_tuple

    def write_basic(self, hidden_layer):
        # map RNN hidden state to image
        # with tf.variable_scope("write", reuse=self.share_parameters):
        decoded_image_portion = dense(hidden_layer,
                                      self.n_hidden, self.img_size * self.img_size * self.num_channels,
                                      scope_name="write")
        # decoded_image_portion = tf.reshape(decoded_image_portion, [-1, self.img_size, self.img_size, self.num_colors])
        return decoded_image_portion

    def write_attention(self, hidden_layer, img_size):
        # with tf.variable_scope("write_attention", reuse=self.share_parameters):
        w = dense(hidden_layer, self.n_hidden, self.attention_n * self.attention_n * self.num_channels,
                  scope_name="write_attention")

        w = tf.reshape(w, [self.attention_n, self.attention_n, self.num_channels])
        w_t = tf.transpose(w, perm=[2, 0, 1])
        Fx, Fy, gamma = self.attn_window("write", hidden_layer, img_size)

        # color1, color2, color3, color1, color2, color3, etc.
        # w_array = tf.reshape(w_t, [self.num_channels * self.batch_size, self.attention_n, self.attention_n])
        w_array = w_t       # [self.num_channels, self.attention_n, self.attention_n]
        Fx_array = tf.stack([Fx, Fx, Fx])
        Fy_array = tf.stack([Fy, Fy, Fy])

        Fyt = tf.transpose(Fy_array, perm=[0, 2, 1])
        # [vert, attn_n] * [attn_n, attn_n] * [attn_n, horiz]
        wr = tf.matmul(Fyt, tf.matmul(w_array, Fx_array))       # [self.num_channels, img_size[0], img_size[1]]

        # sep_colors = tf.reshape(wr, [self.batch_size, self.num_channels, self.img_size ** 2])
        # wr = tf.reshape(wr, [self.num_channels, self.batch_size, self.img_size, self.img_size])
        wr = tf.transpose(wr, [1, 2, 0])
        # wr = tf.reshape(wr, [img_size[0] * img_size[1] * self.num_channels])    # [img_size[0] * img_size[1] * self.num_channels]
        return wr * 1.0/gamma


    def get_training_data(self):
        data = []
        image_label_path = "./data/labels/image_labels_debug.txt" if self.DEBUG else "./data/labels/image_labels.txt"
        with open(image_label_path, "r") as file:
            for line in file:
                image_label_pattern = re.compile(r'(\S+) (\d+) \S+')
                m = image_label_pattern.match(line)
                data.append((m.group(1), int(m.group(2))))       # (filename, label_num)

        data_len = len(data)
        index_list = range(data_len)
        training_data_index = random.sample(index_list, int(self.portion_as_training_data * data_len))
        testing_data_index = list(set(index_list) - set(training_data_index))

        # filename_pattern = re.compile(r'[^/\\]+\.jpg')
        with open("./data/test/test_file_list", "w") as file:
            for i in testing_data_index:
                # filename = filename_pattern.search(data[i]).group()
                file.write(data[i][0] + ' ' + str(data[i][1]) + '\n')

        return [data[i] for i in training_data_index]       # [(filename, label_num)]


    def train_encoder(self):
        data = self.get_training_data()
        # base: first 64 images of the training set
        # base = np.array([get_image(sample_file) for sample_file in data[0:64]])
        # base += 1
        # base /= 2

        # merge the first 64 images to an 8*8 image
        # merge_color() doesn't work on variable size image dataset
        # save_image("results/base.jpg", merge_color(base, [8, 8]))     # 0 <= each pixel <= 1?

        data_len = len(data)
        num_batch = data_len // self.batch_size + (1 if data_len % self.batch_size != 0 else 0)
        # print(data_len // self.batch_size)

        variables_not_initialized = True

        train_writer = tf.summary.FileWriter('./log/train_log')
        test_writer = tf.summary.FileWriter('./log/train_log')

        # TODO: tensorboard functions
        # TODO: might as well concat all images and add padding
        print("graph construction:")
        for e in range(10):
            print("epoch: %i" % e)
            # epoch
            # why skipping 2 batches?
            # for i in range((len(data) / self.batch_size) - 2):
            for batch_id in range(num_batch):
                print("\tbatch id: %i" % batch_id)
                # i: batch number
                batch_upper_bound = (batch_id+1)*self.batch_size if batch_id != data_len / self.batch_size + 1 else data_len
                batch_files = data[batch_id*self.batch_size: batch_upper_bound]
                # [
                #   [batch: tensor[height, width, channels)]
                #   [batch: int]
                # ]
                batch = [[get_image(os.path.join("./data/train", batch_file[0] + ".jpg"), desired_type=tf.float32) for batch_file in batch_files]]
                batch.append([batch_file[1] for batch_file in batch_files])
                # print(self.sess.run(batch[0][0]))
                # batch = tf.stack(batch)        # [batch, height, width, channels]

                # batch_images = np.array(batch).astype(np.float32)
                # batch_images += 1
                # batch_images /= 2
                # self.images = batch_images      # no need to feed anymore







                # reset these variables
                # canvas: list of tensors [batch_size, time, tensor[canvas_size]]
                self.canvas = []
                # mu, logsigma, sigma: [(less or equal than) batch_size, self.sequence_length]
                self.mu, self.logsigma, self.sigma = [], [], []

                self.attn_params = []
                self.generation_loss = []


                # flatten w.r.t #images
                # enc_state_history = tf.unstack(enc_state_history)
                # dec_state_history = tf.unstack(dec_state_history)

                batch_images = batch[0]

                # have to use the whole tuple because of state_is_tuple limitation
                # [batch_len, sequence_length + 1]
                # VICIOUS TRAP!!! the first form copies the same list object many times!
                # see https://stackoverflow.com/questions/5280799/list-append-changing-all-elements-to-the-appended-item
                # enc_state_history = [[self.lstm_enc.zero_state(1, tf.float32)]] * len(batch_images)
                enc_state_history = [[self.lstm_enc.zero_state(1, tf.float32)] for i in range(len(batch_images))]
                # print(enc_state_history[0])
                dec_state_history = [[self.lstm_enc.zero_state(1, tf.float32)] for i in range(len(batch_images))]
                # print(self.sess.run(tf.shape(dec_state_history[0])))

                # h_dec_prev: decoder state of previous time step, list of tensors
                # initialize previous decoder state
                # h_dec_prev = [tf.zeros((self.n_hidden))] * len(batch_images)

                # c_prev: list with shape [batch_size], each element is a tensor of an image canvas
                c_prev = []

                # computation graph construction
                for t in range(self.sequence_length):
                    print("\t\ttimestep: %i" % t)
                    # generate one computation graph for each image? see start of class def
                    # batch_images = tf.unstack(self.images)
                    # x_hat: list of tensors
                    x_hat = []
                    # r: list of tensors
                    r = []
                    # z: list of tensors
                    z = [0 for i in range(len(batch_images))]

                    # i0 = tf.constant(0)

                    for i in range(len(batch_images)):
                        print("\t\t\timage: %i" % i)
                    # def while_body(i):
                        # for each image:
                        # error image + original image
                        batch_image_shape = tf.shape(batch_images[i])

                        if t == 0:
                            # initialize canvas history
                            # list of tensor with size[(less or equal than)batch_size],
                            # with size of each element identical to current image
                            c_prev.append(tf.zeros([batch_image_shape[0], batch_image_shape[1], batch_image_shape[2]]))

                        else:
                            c_prev[i] = self.canvas[i][t - 1]

                        # TODO: no sigmoid? cuz that suppresses gradient
                        # x_hat.append(batch_images[i] - tf.sigmoid(c_prev[i]))
                        x_hat.append(batch_images[i] - c_prev[i])
                        # read the image
                        # r = self.read_basic(x,x_hat,h_dec_prev)
                        # [1, 2 * self.attention_n * self.attention_n * channels]
                        r.append(self.read_attention(batch_images[i], x_hat[i], dec_state_history[i][t][0]))   # use cell state
                        # encode it to gauss distrib

                        # use cell state
                        new_mu, new_logsigma, new_sigma, new_enc_state_tuple = self.encode(enc_state_history[i][t], tf.concat([r[i], dec_state_history[i][t][0]], 1))
                        # mu and sigma: [self.n_z] latent code length

                        if t == 0:
                            # initialize gaussian latent dist history
                            self.mu.append([new_mu])
                            self.logsigma.append([new_logsigma])
                            self.sigma.append([new_sigma])
                        else:
                            # t-th time step of image #i
                            self.mu[i].append(new_mu)
                            self.logsigma[i].append(new_logsigma)
                            self.sigma[i].append(new_sigma)

                        # self.mu[i][t], self.logsigma[i][t], self.sigma[i][t], new_enc_state_tuple = self.encode(enc_state_history, tf.concat(1, [r[i], h_dec_prev[i]]))
                        enc_state_history[i].append(new_enc_state_tuple)  # per image

                        # sample from the distrib to get z
                        # TODO: further research, dont' quite understand
                        # print(t, i)
                        # print(self.mu[t][i])
                        # print(self.sigma[t][i])
                        z[i] = self.sampleQ(self.mu[i][t], self.sigma[i][t], i)  # [self.n_z]: latent variable
                        # retrieve the hidden layer of RNN
                        new_dec_state_tuple = self.decode_layer(dec_state_history[i][t], z[i])
                        # h_dec, new_dec_state_tuple = self.decode_layer(dec_state_history[i], z[i])
                        dec_state_history[i].append(new_dec_state_tuple)
                        # map from hidden layer -> image portion, and then write it.
                        # self.cs[t] = c_prev + self.write_basic(h_dec)

                        # TODO: write window will be extrapolated to canvas size, but ain't it supposed to write a small batch?
                        if t == 0:
                            # initialize canvas
                            self.canvas.append([c_prev[i] + self.write_attention(new_dec_state_tuple.c, tf.shape(batch_images[i]))])

                        else:
                            self.canvas[i].append(c_prev[i] + self.write_attention(new_dec_state_tuple.c, tf.shape(batch_images[i])))

                        # h_dec_prev[i] = h_dec

                        # self.share_parameters = True  # share variables after the first loop
                        # return [i + 1]

                    # tf.while_loop(while_cond, while_body, loop_vars=[i0])

                canvas_list = []
                generation_loss_list = []
                kl_terms_list = []

                # checkout https://stackoverflow.com/questions/35330117/how-can-i-run-a-loop-with-a-tensor-as-its-range-in-tensorflow
                # i0 = tf.constant(0)

                # def while_cond(i, c, l):
                #     return i < tf.shape(self.canvas)[0]  # for all images in a batch

                # def while_body(i, c, l):
                for canvas_index in range(len(self.canvas)):
                    canvas_list.append(self.canvas[canvas_index][-1])  # final canvas per image

                    # TODO: better measure
                    generation_loss_list.append(tf.nn.l2_loss(batch_images[canvas_index] - self.canvas[canvas_index][-1]))  # error image per image
                    # return [i + 1, c, l]

                # canvas_list, generation_loss_list = tf.while_loop(while_cond, while_body, loop_vars=[canvas_list, generation_loss_list])
                # tf.while_loop(while_cond, while_body, loop_vars=[i0, canvas_list, generation_loss_list])

                # for i in range(tf.shape(self.canvas)[0]):
                #     canvas_list.append(self.canvas[i][-1])     # final canvas per image
                #     generation_loss_list.append(tf.nn.l2_loss(self.images[i] - self.canvas[i][-1]))    # error image per image

                # the final timestep
                # TODO: why sigmoid? Pixel range
                # canvas shape: [batch, height, width, channel]
                # self.generated_images = tf.nn.sigmoid(np.array([c[-1] for c in self.canvas]))
                # self.generated_images = tf.nn.sigmoid(canvas_list)
                self.generated_images = canvas_list

                # log likelihood of binary image
                # self.generation_loss = tf.reduce_mean(-tf.reduce_sum(self.images * tf.log(1e-10 + self.generated_images) + (1-self.images) * tf.log(1e-10 + 1 - self.generated_images), 1))
                # TODO: mean or sum?
                self.generation_loss = tf.reduce_mean(tf.stack(generation_loss_list))

                kl_terms = [0 for i in range(self.sequence_length)]
                for t in range(self.sequence_length):
                    # def while_cond(i, mu_list, sigma_list, logsigma_list, kl_terms_list):
                    #     return i < tf.shape(mu_list)[0]

                    # def while_body(i, mu_list, sigma_list, logsigma_list, kl_terms_list):
                    for i in range(len(self.mu)):
                        mu2 = tf.square(self.mu[i][t])
                        sigma2 = tf.square(self.sigma[i][t])
                        logsigma = self.logsigma[i][t]
                        kl_terms_list.append(
                            0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - self.sequence_length * 0.5)

                        # return [i + 1, mu_list, sigma_list, logsigma_list, kl_terms_list]

                    # i0 = tf.constant(0)
                    # _, _, _, _, kl_terms_list = tf.while_loop(while_cond, while_body,
                    #                                           loop_vars=[i0, self.mu, self.sigma, self.logsigma, []])

                    kl_terms_batch_vector = tf.stack(kl_terms_list)  # [batch_size]
                    kl_terms[t] = kl_terms_batch_vector  # [batch_size]
                    kl_terms_list = []

                self.latent_loss = tf.reduce_mean(tf.add_n(kl_terms))


                # classification
                # take encoder state sequence of every image
                batch_encoder_states = []
                # print(len(enc_state_history[0]))
                for i in range(len(batch_images)):
                    # tensor[1, sequence_length * self.n_hidden]
                    batch_encoder_states.append(tf.concat([s.c for s in enc_state_history[i][1:]], axis=1))
                    # print(self.sess.run(tf.shape(batch_encoder_states[i])))

                # tensor[batch_len, self.sequence_length * self.n_hidden]
                batch_encoder_states = tf.concat(batch_encoder_states, axis=0)

                # tensor[batch_len, self.num_class]
                logits = dense(batch_encoder_states, self.sequence_length * self.n_hidden, self.num_class, "pre_softmax")
                labels = tf.constant(batch[1], dtype=tf.int64, name="batch_image_labels")

                prediction_correctness = tf.equal(tf.argmax(logits, axis=1), labels)
                classification_accuracy = tf.reduce_mean(tf.cast(prediction_correctness, tf.float32))

                # tensor[batch_len, self.num_class]
                # labels = tf.one_hot(indices=batch[1], depth=self.num_class, on_value=1.0, off_value=0.0, axis=-1)

                # cross_entropy_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, dim=-1, name=)
                cross_entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,  name="softmax_xentropy")
                classification_loss = tf.reduce_sum(cross_entropy_losses)



                self.cost = self.generation_loss + self.latent_loss + classification_loss
                print("I'm computing gradients!")
                grads = self.optimizer.compute_gradients(self.cost)

                # clip gradients
                for i, (g, v) in enumerate(grads):
                    if g is not None:
                        grads[i] = (tf.clip_by_norm(g, 5), v)
                self.train_op = self.optimizer.apply_gradients(grads)

                # update graph
                train_writer.add_graph(self.sess.graph)
                test_writer.add_graph(self.sess.graph)
                print("I'm done computing gradients!")



                if variables_not_initialized:
                    # initialize variables once
                    print("I'm initializing variables!")
                    self.sess.run(tf.global_variables_initializer())
                    print("I've passed variable init!")
                    saver = tf.train.Saver(max_to_keep=5)
                    variables_not_initialized = False





                # cs, attn_params, gen_loss, lat_loss, _ = self.sess.run([
                #     self.canvas, self.attn_params, self.generation_loss, self.latent_loss,
                #     classification_loss, classification_accuracy, self.train_op
                # ])
                print("I'm running!")
                cs, gen_loss, lat_loss, cls_loss, acc, _ = self.sess.run([
                    self.canvas, self.generation_loss, self.latent_loss,
                    classification_loss, classification_accuracy, self.train_op
                ])
                print("epoch %d batch %d: gen_loss %f, lat_loss %f, classification_loss %f, acc %f"
                      % (e, batch_id, gen_loss, lat_loss, cls_loss, acc))
                # print(attn_params[0].shape)
                # print(attn_params[1].shape)
                # print(attn_params[2].shape)
                ckpt_step_len = 2 if self.DEBUG else 800
                num_demo_image = self.batch_size if self.DEBUG else 10

                if batch_id % ckpt_step_len == 0:

                    saver.save(self.sess, "./model/model", global_step=e*10000 + batch_id)

                    # cs = 1.0/(1.0+np.exp(-np.array(cs)))    # x_recons=sigmoid(canvas)

                    for cs_iter in range(num_demo_image):       # print first 10 images in canvas
                        img = cs[cs_iter][-1]
                        # print(type(img))
                        # results_square = np.reshape(results, [-1, self.img_size, self.img_size, self.num_channels])
                        # print(results_square.shape)
                        # TODO: currently image tensor is clipped to range of [-1, 1], but there could be better ways
                        # to let the network produce pixels of correct range.
                        save_image("results/epoch#" + str(e) + "-batch#" + str(batch_id) + "-iter#" + str(cs_iter) + ".jpg",
                                   numpy.clip(img, a_min=-1, a_max=1))


    # def load_images(self, path, pattern):
    #     data = glob(os.path.join(path, pattern))
    #     images = [get_image(file) for file in data]  # [batch, height, width, channels]
    #     images = np.array(images).astype(np.float32)
    #     self.images = images  # no need to feed anymore


    # TODO: resume/continue training?


    def encode_data(self, data):
        # returns sequence of latent code
        return



    def train_classifier(self):
        return


    def view(self):
        data = glob(os.path.join("./data/train", "*.jpg"))          # TODO: what is that?
        base = np.array([get_image(sample_file, desired_type=tf.float32) for sample_file in data[0:64]])
        base += 1
        base /= 2
        # self.images = base

        save_image("results/base.jpg", merge_color(base, [8, 8]))

        saver = tf.train.Saver(max_to_keep=2)
        saver.restore(self.sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))

        cs, attn_params, gen_loss, lat_loss = self.sess.run([self.canvas, self.attn_params, self.generation_loss, self.latent_loss])
        print("genloss %f latloss %f" % (gen_loss, lat_loss))

        cs = 1.0/(1.0+np.exp(-np.array(cs))) # x_recons=sigmoid(canvas)

        print(np.shape(cs))
        print(np.shape(attn_params))
            # cs[0][cent]

        for cs_iter in range(10):
            results = cs[cs_iter]
            results_square = np.reshape(results, [-1, self.img_size, self.img_size, self.num_channels])

            print(np.shape(results_square))

            for i in range(64):
                center_x = int(attn_params[cs_iter][0][i][0])
                center_y = int(attn_params[cs_iter][1][i][0])
                distance = int(attn_params[cs_iter][2][i][0])

                size = 2

                # for x in range(3):
                #     for y in range(3):
                #         nx = x - 1;
                #         ny = y - 1;
                #
                #         xpos = center_x + nx*distance
                #         ypos = center_y + ny*distance
                #
                #         xpos2 = min(max(0, xpos + size), 63)
                #         ypos2 = min(max(0, ypos + size), 63)
                #
                #         xpos = min(max(0, xpos), 63)
                #         ypos = min(max(0, ypos), 63)
                #
                #         results_square[i,xpos:xpos2,ypos:ypos2,0] = 0;
                #         results_square[i,xpos:xpos2,ypos:ypos2,1] = 1;
                #         results_square[i,xpos:xpos2,ypos:ypos2,2] = 0;
                # print("%f , %f" % (center_x, center_y)

            print(results_square)

            save_image("results/view-clean-step-" + str(cs_iter) + ".jpg", merge_color(results_square, [8, 8]))




model = Draw()
model.train_encoder()
# model.view()
