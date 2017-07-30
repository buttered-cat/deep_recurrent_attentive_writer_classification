import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from glob import glob
import os


# TODO: validation + early stopping. Maybe hold-out + cross validation the rest?

# problem with the whole program is the framework doesn't support "ragged" tensors well, see
# https://datascience.stackexchange.com/questions/15056/how-to-use-lists-in-tensorflow
# maybe a new set of mathematical notations?
# TODO: need a memory mechanism
class Draw():
    def __init__(self):

        self.img_size = 64
        self.num_channels = 3

        self.attention_n = 5
        self.n_hidden = 256
        self.n_z = 10
        self.sequence_length = 10
        self.batch_size = 64
        self.share_parameters = False

        # TODO: variable image size: maybe classify by size and do batch matrix multiplication
        # checkout https://stackoverflow.com/questions/38966533/different-image-sizes-in-tensorflow-with-batch-size-1
        # and https://gist.github.com/eerwitt/518b0c9564e500b4b50f
        self.images = tf.placeholder(tf.float32, [None, None, None, self.num_channels])       # variable image size
        # self.images = None
        # self.load_images("./data/train", "*.jpg")

        self.e = tf.random_normal((self.batch_size, self.n_z), mean=0, stddev=1) # Qsampler noise

        # What kinda structure? A cell IS A CELL, with vectors as input/output
        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # encoder Op
        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True) # decoder Op

        # self.canvas = [0] * self.sequence_length
        # canvas: of shape [batch_size, time, canvas_size]
        self.canvas = []
        # self.mu, self.logsigma, self.sigma = [0] * self.sequence_length, [0] * self.sequence_length, [0] * self.sequence_length
        self.mu, self.logsigma, self.sigma = [], [], []

        # h_dec_prev: decoder state of previous time step, list of tensors
        h_dec_prev = []
        # h_dec_prev = tf.zeros((self.batch_size, self.n_hidden))

        enc_state = self.lstm_enc.zero_state(tf.shape(self.images), tf.float32)
        dec_state = self.lstm_dec.zero_state(tf.shape(self.images), tf.float32)

        # flatten w.r.t #images
        enc_state = tf.unstack(enc_state)
        dec_state = tf.unstack(dec_state)

        # x = tf.reshape(self.images, [-1, self.img_size*self.img_size*self.num_colors])
        # x = self.images     # [batch, height, width, channel]
        self.attn_params = []
        # self.generated_images = []
        self.generation_loss = []
        for t in range(self.sequence_length):
            # TODO: generate one computation graph for each image? No, euivalent to batch_size = 1. see start of class def
            batch_image_list = tf.unstack(self.images)
            # c_prev: with shape [batch_size, time], each element is a tensor of an image canvas
            c_prev = []
            # x_hat: list of tensors
            x_hat = []
            # r: list of tensors
            r = []
            # z: list of tensors
            z = []

            # below defines computation graph
            i0 = tf.constant(0)

            # for i in range(tf.shape(self.images)[0]):
            def while_cond(i):      # for all images in a batch
                return i < tf.shape(self.images)[0]

            def while_body(i):
                # for each image:
                # error image + original image
                batch_image_list_shape = tf.shape(batch_image_list[i])

                x_hat.append(batch_image_list[i] - tf.sigmoid(c_prev[i]))
                # read the image
                # r = self.read_basic(x,x_hat,h_dec_prev)
                r.append(self.read_attention(batch_image_list[i], x_hat[i], h_dec_prev[i]))     # [self.attention_n, 2 * self.attention_n]
                # encode it to gauss distrib

                new_mu, new_logsigma, new_sigma, new_enc_state = self.encode(enc_state,
                                                                             tf.concat(1, [r[i], h_dec_prev[i]]))

                if t == 0:
                    # initialize canvas history
                    c_prev.append([tf.zeros((batch_image_list_shape[0] * batch_image_list_shape[1]))])

                    # initialize gaussian latent dist history
                    self.mu.append([new_mu])
                    self.logsigma.append([new_logsigma])
                    self.sigma.append([new_sigma])
                else:
                    # t-th time step of image #i
                    self.mu[t].append(new_mu)
                    self.logsigma[t].append(new_logsigma)
                    self.sigma[t].append(new_sigma)

                    c_prev[i].append(self.canvas[i][t - 1])

                # self.mu[i][t], self.logsigma[i][t], self.sigma[i][t], new_enc_state = self.encode(enc_state, tf.concat(1, [r[i], h_dec_prev[i]]))
                enc_state[i].assign(new_enc_state)

                # sample from the distrib to get z
                # TODO: further research, dont' quite understand
                z[i] = self.sampleQ(self.mu[i][t], self.sigma[i][t])
                # retrieve the hidden layer of RNN
                h_dec, new_dec_state = self.decode_layer(dec_state[i], z[i])
                dec_state[i].assign(new_dec_state)
                # map from hidden layer -> image portion, and then write it.
                # self.cs[t] = c_prev + self.write_basic(h_dec)

                if t == 0:
                    # initialize canvas
                    self.canvas.append([c_prev[i] + self.write_attention(h_dec)])

                    # initialize previous decoder state
                    h_dec_prev.append(h_dec)
                else:
                    self.canvas[i].append(c_prev[i] + self.write_attention(h_dec))
                    h_dec_prev[i] = h_dec

                tf.while_loop(while_cond, while_body, loop_vars=[i])

            self.share_parameters = True  # from now on, share variables

        canvas_list = []
        generation_loss_list = []

        # check https://stackoverflow.com/questions/35330117/how-can-i-run-a-loop-with-a-tensor-as-its-range-in-tensorflow
        i0 = tf.constant(0)

        def while_cond(i, c, l):
            return i < tf.shape(self.canvas)[0]      # for all images in a batch

        def while_body(i, c, l):
            c.append(self.canvas[i][-1])     # final canvas per image
            l.append(tf.nn.l2_loss(self.images[i] - self.canvas[i][-1]))    # error image per image
            return [i + 1, c, l]

        # canvas_list, generation_loss_list = tf.while_loop(while_cond, while_body, loop_vars=[canvas_list, generation_loss_list])
        tf.while_loop(while_cond, while_body, loop_vars=[i0, canvas_list, generation_loss_list])

        # for i in range(tf.shape(self.canvas)[0]):
        #     canvas_list.append(self.canvas[i][-1])     # final canvas per image
        #     generation_loss_list.append(tf.nn.l2_loss(self.images[i] - self.canvas[i][-1]))    # error image per image

        # the final timestep
        # TODO: why sigmoid?
        # canvas shape: [batch, height, width, channel]
        # self.generated_images = tf.nn.sigmoid(np.array([c[-1] for c in self.canvas]))
        self.generated_images = tf.nn.sigmoid(canvas_list)

        # log likelihood of binary image
        # TODO: better measure
        # self.generation_loss = tf.reduce_mean(-tf.reduce_sum(self.images * tf.log(1e-10 + self.generated_images) + (1-self.images) * tf.log(1e-10 + 1 - self.generated_images), 1))
        # TODO: mean or sum?
        self.generation_loss = tf.reduce_sum(generation_loss_list)

        kl_terms = [0]*self.sequence_length
        for t in range(self.sequence_length):
            mu2 = tf.square(self.mu[t])
            sigma2 = tf.square(self.sigma[t])
            logsigma = self.logsigma[t]
            kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2*logsigma, 1) - self.sequence_length*0.5
        self.latent_loss = tf.reduce_mean(tf.add_n(kl_terms))
        self.cost = self.generation_loss + self.latent_loss
        optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
        grads = optimizer.compute_gradients(self.cost)
        for i,(g,v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g,5),v)
        self.train_op = optimizer.apply_gradients(grads)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    # given a hidden decoder layer:
    # locate where to put attention filters
    # TODO: maybe variable window aspect ratio?
    def attn_window(self, scope, h_dec, img_shape):
        # with tf.variable_scope(scope, reuse=self.share_parameters):
        parameters = dense(h_dec, self.n_hidden, 5, scope=scope, reuse_params=self.share_parameters)     # [batch, 5]
        # gx_, gy_: center of 2d gaussian on a scale of -1 to 1
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(1, 5, parameters)   # each: [1(batch), 1]

        # move gx/gy to be a scale of -imgsize to +imgsize (?)
        # dense() doesn't seem to guarantee positiveness, but that's not a problem. See paper
        # "The scaling...is chosen to ensure that the initial patch...roughly covers the whole image."
        gx = (img_shape[1]+1)/2 * (gx_ + 1)
        gy = (img_shape[0]+1)/2 * (gy_ + 1)

        sigma2 = tf.exp(log_sigma2)
        # stride/delta: how far apart these patches will be
        # typo?
        # delta = (self.img_size - 1) / ((self.attention_n-1) * tf.exp(log_delta))
        delta = (tf.maximum(img_shape[0], img_shape[1]) - 1) / (self.attention_n-1) * tf.exp(log_delta)
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
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), 1e-8)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), 1e-8)
        return Fx, Fy       # [self.attention_n, img_size[1 | 0]]


    # the read() operation without attention
    def read_basic(self, x, x_hat, h_dec_prev):
        return tf.concat(1,[x,x_hat])

    def read_attention(self, x, x_hat, h_dec_prev):
        # per image
        # TODO: single image
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
            img_t = tf.transpose(img, perm=[3, 0, 1])   # [channel, height, width]

            # batch_colors_array = tf.reshape(img_t, [self.num_channels * self.batch_size, self.img_size, self.img_size])
            batch_colors_array = img_t
            Fx_array = tf.concat(0, [Fx, Fx, Fx])       # 3 channels
            Fy_array = tf.concat(0, [Fy, Fy, Fy])

            Fxt = tf.transpose(Fx_array, perm=[0,2,1])      # transpose per image

            # Apply the gaussian patches:
            # square patch
            # TODO: variable aspect ratio glimpse?
            glimpse = tf.batch_matmul(Fy_array, tf.batch_matmul(batch_colors_array, Fxt))   # tensor mul
            # glimpse = tf.reshape(glimpse, [self.num_channels, self.batch_size, self.attention_n, self.attention_n])
            glimpse = tf.transpose(glimpse, [1, 2, 0])      # [height, width, channel]
            # reshape: iterate through two tensors simultaneously, and fill the elements
            # glimpse = tf.reshape(glimpse, [self.batch_size, self.attention_n * self.attention_n * self.num_channels])
            glimpse = tf.reshape(glimpse, [1, -1])      # [1, height * width * channel]
            # finally scale this glimpse with the gamma parameter
            return glimpse * tf.reshape(gamma, [-1, 1])
        x = filter_img(x, Fx, Fy, gamma)            # [self.attention_n, self.attention_n]
        x_hat = filter_img(x_hat, Fx, Fy, gamma)
        return tf.concat(1, [x, x_hat])        # [self.attention_n, 2 * self.attention_n]

    # encode an attention patch
    def encode(self, prev_state, image):
        # update the RNN with image
        with tf.variable_scope("encoder",reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_enc(image, prev_state)

        # map the RNN hidden state to latent variables
        # TODO: latent bug? dense() modifies scope if not provided
        # with tf.variable_scope("mu", reuse=self.share_parameters):
        mu = dense(hidden_layer, self.n_hidden, self.n_z, scope="mu", reuse_params=self.share_parameters)
        # with tf.variable_scope("sigma", reuse=self.share_parameters):
        logsigma = dense(hidden_layer, self.n_hidden, self.n_z, scope="sigma", reuse_params=self.share_parameters)
        with tf.variable_scope("sigma", reuse=self.share_parameters):
            sigma = tf.exp(logsigma)
        return mu, logsigma, sigma, next_state


    def sampleQ(self, mu, sigma):
        return mu + sigma*self.e

    def decode_layer(self, prev_state, latent):
        # update decoder RNN with latent var
        with tf.variable_scope("decoder", reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_dec(latent, prev_state)

        return hidden_layer, next_state

    def write_basic(self, hidden_layer):
        # map RNN hidden state to image
        # with tf.variable_scope("write", reuse=self.share_parameters):
        decoded_image_portion = dense(hidden_layer,
                                      self.n_hidden, self.img_size * self.img_size * self.num_channels,
                                      scope="write",
                                      reuse_params=self.share_parameters)
        # decoded_image_portion = tf.reshape(decoded_image_portion, [-1, self.img_size, self.img_size, self.num_colors])
        return decoded_image_portion

    def write_attention(self, hidden_layer):
        # with tf.variable_scope("write_attention", reuse=self.share_parameters):
        w = dense(hidden_layer, self.n_hidden, self.attention_n * self.attention_n * self.num_channels,
                  scope="write_attention", reuse_params=self.share_parameters)

        w = tf.reshape(w, [self.batch_size, self.attention_n, self.attention_n, self.num_channels])
        w_t = tf.transpose(w, perm=[3,0,1,2])
        Fx, Fy, gamma = self.attn_window("write", hidden_layer)

        # color1, color2, color3, color1, color2, color3, etc.
        w_array = tf.reshape(w_t, [self.num_channels * self.batch_size, self.attention_n, self.attention_n])
        Fx_array = tf.concat(0, [Fx, Fx, Fx])
        Fy_array = tf.concat(0, [Fy, Fy, Fy])

        Fyt = tf.transpose(Fy_array, perm=[0,2,1])
        # [vert, attn_n] * [attn_n, attn_n] * [attn_n, horiz]
        wr = tf.batch_matmul(Fyt, tf.batch_matmul(w_array, Fx_array))
        sep_colors = tf.reshape(wr, [self.batch_size, self.num_channels, self.img_size ** 2])
        wr = tf.reshape(wr, [self.num_channels, self.batch_size, self.img_size, self.img_size])
        wr = tf.transpose(wr, [1,2,3,0])
        wr = tf.reshape(wr, [self.batch_size, self.img_size * self.img_size * self.num_channels])
        return wr * tf.reshape(1.0/gamma, [-1, 1])


    def train(self):
        data = glob(os.path.join("./data/train", "*.jpg"))
        base = np.array([get_image(sample_file) for sample_file in data[0:64]])     # TODO: what does base do?
        base += 1
        base /= 2

        ims("results/base.jpg",merge_color(base,[8,8]))

        saver = tf.train.Saver(max_to_keep=2)

        for e in range(10):
            # epoch
            for i in range((len(data) / self.batch_size) - 2):

                batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = [get_image(batch_file) for batch_file in batch_files]   # [batch, height, width, channels]
                batch_images = np.array(batch).astype(np.float32)
                batch_images += 1
                batch_images /= 2
                # self.images = batch_images      # no need to feed anymore

                cs, attn_params, gen_loss, lat_loss, _ = self.sess.run([self.canvas, self.attn_params, self.generation_loss, self.latent_loss, self.train_op], feed_dict={self.images: batch_images})
                print("epoch %d iter %d genloss %f latloss %f" % (e, i, gen_loss, lat_loss))
                # print(attn_params[0].shape)
                # print(attn_params[1].shape)
                # print(attn_params[2].shape)
                if i % 800 == 0:

                    saver.save(self.sess, os.getcwd() + "/training/train", global_step=e*10000 + i)

                    cs = 1.0/(1.0+np.exp(-np.array(cs))) # x_recons=sigmoid(canvas)

                    for cs_iter in range(10):
                        results = cs[cs_iter]
                        results_square = np.reshape(results, [-1, self.img_size, self.img_size, self.num_channels])
                        print(results_square.shape)
                        ims("results/"+str(e)+"-"+str(i)+"-step-"+str(cs_iter)+".jpg",merge_color(results_square,[8,8]))


    # def load_images(self, path, pattern):
    #     data = glob(os.path.join(path, pattern))
    #     images = [get_image(file) for file in data]  # [batch, height, width, channels]
    #     images = np.array(images).astype(np.float32)
    #     self.images = images  # no need to feed anymore


    def view(self):
        data = glob(os.path.join("./data/train", "*.jpg"))          # TODO: what is that?
        base = np.array([get_image(sample_file) for sample_file in data[0:64]])
        base += 1
        base /= 2
        # self.images = base

        ims("results/base.jpg",merge_color(base,[8,8]))

        saver = tf.train.Saver(max_to_keep=2)
        saver.restore(self.sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))

        cs, attn_params, gen_loss, lat_loss = self.sess.run([self.canvas, self.attn_params, self.generation_loss, self.latent_loss], feed_dict={self.images: base})
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

                size = 2;

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

            ims("results/view-clean-step-"+str(cs_iter)+".jpg",merge_color(results_square,[8,8]))




model = Draw()
# model.train()
model.view()
