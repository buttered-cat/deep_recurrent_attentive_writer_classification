import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from constants import *
from glob import glob
import os
import random
import re
import subprocess


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
        self.DEBUG = False

        self.img_size = 64
        self.num_channels = 3

        self.attention_n = 24
        self.n_hidden = 128
        self.n_z = 50       # latent code length
        self.num_class = 134        # number of label classes
        self.sequence_length = 3 if self.DEBUG else 12
        self.batch_size = 2 if self.DEBUG else 8
        self.portion_as_training_data = 4/5
        self.share_parameters = False

        self.optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)

        # TODO: variable image size: maybe classify by size and do batch matrix multiplication
        # checkout https://stackoverflow.com/questions/38966533/different-image-sizes-in-tensorflow-with-batch-size-1
        # and https://gist.github.com/eerwitt/518b0c9564e500b4b50f
        # tensor[batch_len, max(image height), max(image width), channels]
        self.images = tf.placeholder(tf.float32, [None, None, None, self.num_channels])       # variable image size
        self.labels = tf.placeholder(tf.int64, [None])

        # Qsampler noise
        self.e = tf.random_normal([self.batch_size, self.n_z], mean=0, stddev=1)   # tensor[self.batch_size, self.n_z]

        # What kinda structure? A cell IS A CELL, with vectors as input/output
        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # encoder Op
        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # decoder Op

        # canvas: list of tensors [batch_size, time, tensor[canvas_size]]
        self.canvas = [0] * self.sequence_length
        # mu, logsigma, sigma: [(less or equal than) batch_size, self.sequence_length]
        self.mu, self.logsigma, self.sigma = [0] * self.sequence_length, [0] * self.sequence_length, [0] * self.sequence_length

        # x = tf.reshape(self.images, [-1, self.img_size*self.img_size*self.num_colors])
        # x = self.images     # [batch, height, width, channel]
        self.attn_params = []
        # self.generated_images = []
        self.generation_loss = []

        self.train_writer = tf.summary.FileWriter('./log/train_log')
        self.test_writer = tf.summary.FileWriter('./log/train_log')

        print("GPU state before creating session:")
        print(subprocess.check_output([NVIDIA_SMI_PATH]).decode('utf-8'))

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = tf.Session()

        print("\nGPU state after creating session:")
        print(subprocess.check_output([NVIDIA_SMI_PATH]).decode('utf-8'))

        x = self.images
        batch_shape = tf.shape(x)

        # have to use the whole tuple because of state_is_tuple limitation
        # [sequence_len, tensor[batch_len, self.n_hidden]]
        # VICIOUS TRAP!!! the first form copies the same list object many times!
        # see https://stackoverflow.com/questions/5280799/list-append-changing-all-elements-to-the-appended-item
        # enc_state_history = [[self.lstm_enc.zero_state(1, tf.float32)]] * len(x)
        enc_state_history = [0] * (self.sequence_length + 1)
        dec_state_history = [0] * (self.sequence_length + 1)
        enc_state_history[0] = self.lstm_enc.zero_state(batch_shape[0], tf.float32)
        dec_state_history[0] = self.lstm_enc.zero_state(batch_shape[0], tf.float32)

        # h_dec_prev: decoder state of previous time step, list of tensors
        # initialize previous decoder state
        # h_dec_prev = [tf.zeros((self.n_hidden))] * len(x)

        # c_prev: list with shape [batch_size], each element is a tensor of an image canvas
        c_prev = []

        # computation graph construction
        print("I'm constructing graph!")
        for t in range(self.sequence_length):
            print("\ttimestep: %i" % t)
            # generate one computation graph for each image? see start of class def
            # x = tf.unstack(self.images)
            c_prev = tf.zeros(batch_shape) if t is 0 else self.canvas[t - 1]

            # TODO: no sigmoid? cuz that suppresses gradient
            # x_hat.append(x[i] - tf.sigmoid(c_prev[i]))
            x_hat = x - c_prev
            # read the image
            # [1, 2 * self.attention_n * self.attention_n * channels]
            r = self.read_attention(x, x_hat, dec_state_history[t].c)  # use cell state
            # encode it to gauss distrib
            # use cell state
            self.mu[t], self.logsigma[t], self.sigma[t], enc_state_history[t+1] = self.encode(enc_state_history[t],
                                                                               tf.concat([r, dec_state_history[t].c], 1))

            # sample from the distrib to get z
            # TODO: further research, dont' quite understand
            # print(t, i)
            # print(self.mu[t][i])
            # print(self.sigma[t][i])
            z = self.sampleQ(self.mu[t], self.sigma[t])  # [self.n_z]: latent variable
            # retrieve the hidden layer of RNN
            new_dec_state_tuple = self.decode_layer(dec_state_history[t], z)
            # h_dec, new_dec_state_tuple = self.decode_layer(dec_state_history[i], z[i])
            dec_state_history[t+1] = new_dec_state_tuple
            # map from hidden layer -> image portion, and then write it.
            # self.cs[t] = c_prev + self.write_basic(h_dec)

            # TODO: write window will be extrapolated to canvas size, but ain't it supposed to write a small patch?
            self.canvas[t] = c_prev + self.write_attention(new_dec_state_tuple.c, batch_shape)
            self.share_parameters = True # from now on, share variables


        canvas_list = []
        generation_loss_list = []
        kl_terms_list = []

        # checkout https://stackoverflow.com/questions/35330117/how-can-i-run-a-loop-with-a-tensor-as-its-range-in-tensorflow
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
        self.generated_images = self.canvas[-1]

        # log likelihood of binary image
        # self.generation_loss = tf.reduce_mean(-tf.reduce_sum(self.images * tf.log(1e-10 + self.generated_images) + (1-self.images) * tf.log(1e-10 + 1 - self.generated_images), 1))
        # TODO: mean or sum?
        self.generation_loss = tf.nn.l2_loss(x - self.generated_images)

        kl_terms = [0] * self.sequence_length
        for t in range(self.sequence_length):
            mu2 = tf.square(self.mu[t])
            sigma2 = tf.square(self.sigma[t])
            logsigma = self.logsigma[t]
            kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2*logsigma, 1) - self.sequence_length*0.5
        self.latent_loss = tf.reduce_mean(tf.add_n(kl_terms))

        # classification
        # take encoder state sequence of every image
        batch_encoder_states = tf.stack([s.c for s in enc_state_history[1:]])
        batch_encoder_states = tf.transpose(batch_encoder_states, perm=[1, 0, 2])
        batch_encoder_states = tf.reshape(batch_encoder_states, [-1, self.sequence_length * self.n_hidden])
        # print(len(enc_state_history[0]))

        # tensor[batch_len, self.num_class]
        logits = dense(batch_encoder_states, self.sequence_length * self.n_hidden, self.num_class, "pre_softmax")
        labels = self.labels

        prediction_correctness = tf.equal(tf.argmax(logits, axis=1), labels)
        self.classification_accuracy = tf.reduce_mean(tf.cast(prediction_correctness, tf.float32))

        # tensor[batch_len, self.num_class]
        # labels = tf.one_hot(indices=batch[1], depth=self.num_class, on_value=1.0, off_value=0.0, axis=-1)

        # cross_entropy_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, dim=-1, name=)
        cross_entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                              name="softmax_xentropy")
        self.classification_loss = tf.reduce_sum(cross_entropy_losses)

        self.cost = self.generation_loss + self.latent_loss + self.classification_loss
        print("I'm computing gradients!")
        grads = self.optimizer.compute_gradients(self.cost)

        # clip gradients
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)
        self.train_op = self.optimizer.apply_gradients(grads)

        # update graph
        self.train_writer.add_graph(self.sess.graph)
        self.test_writer.add_graph(self.sess.graph)
        print("I'm done computing gradients!")

        print("I'm initializing variables!")
        self.sess.run(tf.global_variables_initializer())
        print("I've passed variable init!")

    # given a hidden decoder layer:
    # locate where to put attention filters
    # TODO: maybe variable window aspect ratio?
    def attn_window(self, scope, h_dec, batch_shape):
        # with tf.variable_scope(scope, reuse=self.share_parameters):
        parameters = dense(h_dec, self.n_hidden, 5, scope_name=scope)     # [batch, 5]
        # gx_, gy_: center of 2d gaussian on a scale of -1 to 1
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(parameters, 5, axis=1)   # each: [batch, 1]

        # move gx/gy to be a scale of -imgsize to +imgsize (?)
        # dense() doesn't seem to guarantee positiveness, but that's not a problem. See paper
        # "The scaling...is chosen to ensure that the initial patch...roughly covers the whole image."
        gx = tf.cast((batch_shape[2]+1)/2, tf.float32) * (gx_ + 1)
        gy = tf.cast((batch_shape[1]+1)/2, tf.float32) * (gy_ + 1)

        sigma2 = tf.exp(log_sigma2)
        # stride/delta: how far apart these patches will be
        # typo?
        # delta = (self.img_size - 1) / ((self.attention_n-1) * tf.exp(log_delta))
        delta = tf.cast((tf.maximum(batch_shape[1], batch_shape[2]) - 1) / (self.attention_n-1), tf.float32) * tf.exp(log_delta)
        # returns [Fx, Fy, gamma]

        gamma = tf.exp(log_gamma)
        self.attn_params.append([gx, gy, delta, sigma2, gamma])

        return self.filterbank(gx, gy, sigma2, delta, batch_shape) + (gamma,)

    # Given a center, distance, and spread
    # Construct [attention_n x attention_n] patches of gaussian filters
    # represented by Fx = horizontal gaussian, Fy = vertical gaussian
    def filterbank(self, gx, gy, sigma2, delta, batch_shape):
        # 1 x N, look like [[0,1,2,3,4]]
        grid_i = tf.reshape(tf.cast(tf.range(self.attention_n), tf.float32), [1, -1])
        # centers for the individual patches
        mu_x = gx + (grid_i - (self.attention_n + 1)/2) * delta     # [batch_len, self.attention_n]
        mu_y = gy + (grid_i - (self.attention_n + 1)/2) * delta
        mu_x = tf.reshape(mu_x, [-1, self.attention_n, 1])          # [batch_len, self.attention_n, 1]
        mu_y = tf.reshape(mu_y, [-1, self.attention_n, 1])
        # 1 x 1 x imgsize, looks like [[[0,1,2,3,4,...,27]]]
        im_x = tf.reshape(tf.cast(tf.range(batch_shape[2]), tf.float32), [1, 1, -1])       # [1, 1, batch_shape[2]]
        im_y = tf.reshape(tf.cast(tf.range(batch_shape[1]), tf.float32), [1, 1, -1])
        # list of gaussian curves for x and y
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])     # [batch, 1, 1]
        Fx = tf.exp(-tf.square((im_x - mu_x) / (2*sigma2)))     # [batch_len, self.attention_n, batch_shape[2]]
        Fy = tf.exp(-tf.square((im_y - mu_y) / (2*sigma2)))
        # normalize so area-under-curve = 1
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), 1e-8)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), 1e-8)
        return Fx, Fy       # [batch_len, self.attention_n, batch_shape[2 | 1]]


    # the read() operation without attention
    def read_basic(self, x, x_hat, h_dec_prev):
        return tf.concat([x, x_hat], 1)

    def read_attention(self, x, x_hat, h_dec_prev):
        Fx, Fy, gamma = self.attn_window("read", h_dec_prev, tf.shape(x))
        # we have the parameters for a patch of gaussian filters. apply them.
        Fxt = tf.transpose(Fx, perm=[0, 2, 1])  # transpose per image
        Fxt = tf.expand_dims(Fxt, 1)
        Fxt = tf.concat([Fxt] * self.num_channels, axis=1)
        Fy = tf.expand_dims(Fy, 1)  # [batch_len, 1, height, width]
        Fy = tf.concat([Fy] * self.num_channels, axis=1)    # [batch_len, 3, height, width]

        def filter_img(img, Fx, Fy, gamma):
            # img: batch * height * width * channels
            # img = tf.reshape(img, [-1, self.img_size, self.img_size, self.num_channels])    # batch_len * height * width * num_channels
            # img_t = tf.transpose(img, perm=[3,0,1,2])   # channel * batch * height * width
            img_t = tf.transpose(img, perm=[0, 3, 1, 2])   # [batch_len, num_channels, height, width]

            # batch_colors_array = tf.reshape(img_t, [self.num_channels * self.batch_size, self.img_size, self.img_size])

            # Apply the gaussian patches:
            # square patch
            # TODO: variable aspect ratio glimpse?
            # [batch_len, num_channels, attention_n, attention_n]
            glimpse = tf.matmul(Fy, tf.matmul(img_t, Fxt))   # tensor mul
            # glimpse = tf.reshape(glimpse, [self.num_channels, self.batch_size, self.attention_n, self.attention_n])
            glimpse = tf.transpose(glimpse, [0, 2, 3, 1])      # [batch_len, attention_n, attention_n, num_channels]
            # reshape: iterate through two tensors simultaneously, and fill the elements
            # glimpse = tf.reshape(glimpse, [self.batch_size, self.attention_n * self.attention_n * self.num_channels])
            # [batch_len, attention_n * attention_n * channels]
            glimpse = tf.reshape(glimpse, [-1, self.attention_n * self.attention_n * self.num_channels])
            # print(self.sess.run(tf.shape(glimpse)))
            # finally scale this glimpse with the gamma parameter
            return glimpse * gamma  # [batch_len, attention_n * attention_n * channels]
        x = filter_img(x, Fx, Fy, gamma)            # [1, self.attention_n * self.attention_n * channels]
        x_hat = filter_img(x_hat, Fx, Fy, gamma)
        # print(self.sess.run(tf.shape(x)))
        # print(self.sess.run(tf.shape(x_hat)))
        return tf.concat([x, x_hat], 1)        # [batch_len, 2 * attention_n * attention_n * channels]

    # encode an attention patch
    def encode(self, prev_state, image):
        # update the RNN with image
        with tf.variable_scope("encoder", reuse=self.share_parameters):
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
        mu = dense(new_state_tuple.c, self.n_hidden, self.n_z, scope_name="mu")   # [batch_len, self.n_z]
        # with tf.variable_scope("sigma", reuse=self.share_parameters):
        logsigma = dense(new_state_tuple.c, self.n_hidden, self.n_z, scope_name="sigma")
        with tf.variable_scope("sigma", reuse=self.share_parameters):
            sigma = tf.exp(logsigma)
        return mu, logsigma, sigma, new_state_tuple


    def sampleQ(self, mu, sigma):
        return mu + sigma*self.e        # element-wise multiplication of two tensors

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

    def write_attention(self, hidden_layer, batch_shape):
        # with tf.variable_scope("write_attention", reuse=self.share_parameters):
        w = dense(hidden_layer, self.n_hidden, self.attention_n * self.attention_n * self.num_channels,
                  scope_name="write_attention")

        w = tf.reshape(w, [-1, self.attention_n, self.attention_n, self.num_channels])
        w_t = tf.transpose(w, perm=[0, 3, 1, 2])
        Fx, Fy, gamma = self.attn_window("write", hidden_layer, batch_shape)

        # color1, color2, color3, color1, color2, color3, etc.
        # w_array = tf.reshape(w_t, [self.num_channels * self.batch_size, self.attention_n, self.attention_n])
        Fyt = tf.transpose(Fy, perm=[0, 2, 1])  # transpose per image
        Fyt = tf.expand_dims(Fyt, 1)
        Fyt = tf.concat([Fyt] * self.num_channels, axis=1)
        Fx = tf.expand_dims(Fx, 1)  # [batch_len, 1, attention_n, width]
        Fx = tf.concat([Fx] * self.num_channels, axis=1)    # [batch_len, 3, attention_n, width]


        wr = tf.matmul(Fyt, tf.matmul(w_t, Fx))       # [batch_len, self.num_channels, batch_shape[1], batch_shape[2]]

        # sep_colors = tf.reshape(wr, [self.batch_size, self.num_channels, self.img_size ** 2])
        # wr = tf.reshape(wr, [self.num_channels, self.batch_size, self.img_size, self.img_size])
        wr = tf.transpose(wr, [0, 2, 3, 1])
        # wr = tf.reshape(wr, [img_size[0] * img_size[1] * self.num_channels])    # [img_size[0] * img_size[1] * self.num_channels]
        gamma = tf.reshape(gamma, [-1, 1, 1, 1])
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


    def generate_batches(self, data):
        data_len = len(data)
        num_batches = data_len // self.batch_size + (1 if data_len % self.batch_size != 0 else 0)
        batches = []

        for batch_id in range(num_batches):
            batch_upper_bound = (batch_id + 1) * self.batch_size if batch_id != data_len / self.batch_size + 1 else data_len
            batch_files = data[batch_id * self.batch_size: batch_upper_bound]
            labels = [batch_file[1] for batch_file in batch_files]
            batches.append((batch_id, batch_files, labels))

        return batches     # [(batch_id, array of (filename, label_num), labels)]

    def generate_batch_tensor(self, batch_files):
        batch = np.asarray([get_image(os.path.join("./data/train", batch_file[0] + ".jpg"))
                            for batch_file in batch_files])

        # https://stackoverflow.com/questions/32037893/numpy-fix-array-with-rows-of-different-lengths-by-filling-the-empty-elements-wi
        # batch_len * height
        widths = np.array([len(img[0]) for img in batch])
        heights = np.array([len(img[:, 0]) for img in batch])

        width_mask = np.arange(widths.max()) < widths[:, None]  # batch_len * max_width
        height_mask = np.arange(heights.max()) < heights[:, None]  # batch_len * max_height

        width_mask = np.expand_dims(width_mask, axis=1)  # batch_len * 1 * max_width
        height_mask = height_mask[:, :, None]  # batch_len * max_height * 1
        mask = np.logical_and(width_mask, height_mask)  # batch_len * max_height * max_width
        mask = np.stack([mask] * self.num_channels, axis=3)  # batch_len * max_height * max_width * 3(channels)

        out = np.zeros(mask.shape, dtype=np.float32)
        out[mask] = np.concatenate([np.reshape(image, (-1)) for image in batch])
        return out


    def train(self):
        data = self.get_training_data()
        # base: first 64 images of the training set
        # base = np.array([get_image(sample_file) for sample_file in data[0:64]])
        # base += 1
        # base /= 2

        # merge the first 64 images to an 8*8 image
        # merge_color() doesn't work on variable size image dataset
        # save_image("results/base.jpg", merge_color(base, [8, 8]))     # 0 <= each pixel <= 1?

        batches = self.generate_batches(data)

        # print(data_len // self.batch_size)

        saver = tf.train.Saver(max_to_keep=5)


        # TODO: tensorboard functions
        for e in range(10):
            print("epoch: %i" % e)
            # epoch
            # why skipping 2 batches?
            # for i in range((len(data) / self.batch_size) - 2):
            for batch_id, batch_images, batch_labels in batches:
                print("\tbatch id: %i" % batch_id)
                # print(self.sess.run(batch[0][0]))
                # batch = tf.stack(batch)        # [batch, height, width, channels]

                # batch_images = np.array(batch).astype(np.float32)
                # batch_images += 1
                # batch_images /= 2
                # self.images = batch_images      # no need to feed anymore

                # cs, attn_params, gen_loss, lat_loss, _ = self.sess.run([
                #     self.canvas, self.attn_params, self.generation_loss, self.latent_loss,
                #     classification_loss, classification_accuracy, self.train_op
                # ])
                print("\tI'm running!")
                batch_images = self.generate_batch_tensor(batch_images)
                cs, gen_loss, lat_loss, cls_loss, acc, _ = self.sess.run([
                    self.canvas, self.generation_loss, self.latent_loss,
                    self.classification_loss, self.classification_accuracy, self.train_op
                ], feed_dict={self.images: batch_images, self.labels: batch_labels})
                print("\tepoch %d batch %d: gen_loss %f, lat_loss %f, classification_loss %f, acc %f"
                      % (e, batch_id, gen_loss, lat_loss, cls_loss, acc))
                del batch_images        # free memory
                print("\tdeleted batch image tensor.")

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
                                   np.clip(img, a_min=-1, a_max=1))


    # def load_images(self, path, pattern):
    #     data = glob(os.path.join(path, pattern))
    #     images = [get_image(file) for file in data]  # [batch, height, width, channels]
    #     images = np.array(images).astype(np.float32)
    #     self.images = images  # no need to feed anymore


    # TODO: resume training?

    def view(self):
        data = glob(os.path.join("./data/train", "*.jpg"))          # TODO: what is that?
        base = np.array([get_image(sample_file) for sample_file in data[0:64]])
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
model.train()
# model.view()
# data = model.get_training_data()
# batches = model.generate_batches(data)
# save_image("data/padded_images/1.jpg", batches[0][1][0, :, :, :])
