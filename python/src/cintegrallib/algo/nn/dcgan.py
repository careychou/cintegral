import tensorflow as tf


class Generator:
    def __init__(self, depths=[1024, 512, 256, 128], channel=3, s_size=4):
        self.depths = depths + [channel]
        self.s_size = s_size
        self.reuse = False

    def __call__(self, inputs, training=False):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('g', reuse=self.reuse):
            # reshape from inputs
            with tf.variable_scope('reshape'):
                outputs = tf.layers.dense(inputs, self.depths[0] * self.s_size * self.s_size)
                outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            # deconvolution (transpose of convolution) x 4
            with tf.variable_scope('deconv1'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv2'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv3'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv4'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
            # output images
            with tf.variable_scope('tanh'):
                outputs = tf.tanh(outputs, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g/')
        return outputs


class Discriminator:
    def __init__(self, depths=[64, 128, 256, 512], channel=3):
        self.depths = [channel] + depths
        self.reuse = False

    def __call__(self, inputs, training=False, name=''):
        outputs_layer = []

        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)

        outputs = tf.convert_to_tensor(inputs)
        outputs_layer.append(outputs)

        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # convolution x 4
            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                outputs_layer.append(outputs)
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                outputs_layer.append(outputs)
            with tf.variable_scope('conv3'):
                outputs = tf.layers.conv2d(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                outputs_layer.append(outputs)
            with tf.variable_scope('conv4'):
                outputs = tf.layers.conv2d(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                outputs_layer.append(outputs)
            with tf.variable_scope('classify'):
                batch_size = outputs.get_shape()[0].value
                reshape = tf.reshape(outputs, [batch_size, -1])
                outputs_layer.append(reshape)
                outputs = tf.layers.dense(reshape, 2, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d/')
        self.outputs_layer = outputs_layer
        return outputs


class DCGAN:
    def __init__(self,
                 batch_size=128, s_size=4, z_dim=100,
                 g_depths=[1024, 512, 256, 128],
                 d_depths=[64, 128, 256, 512],
                 channel=3):
        self.batch_size = batch_size
        self.s_size = s_size
        self.z_dim = z_dim
        self.g = Generator(depths=g_depths, s_size=self.s_size, channel=channel)
        self.d = Discriminator(depths=d_depths, channel=channel)
        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

    def z_latent(self):
        print(self.z)
        return self.z

    def loss(self, traindata):
        generated = self.g(self.z, training=True)
        g_outputs = self.d(generated, training=True, name='g')
        t_outputs = self.d(traindata, training=True, name='t')
        # add each losses to collection
        tf.add_to_collection(
            'g_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=g_outputs)))
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=t_outputs)))
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([self.batch_size], dtype=tf.int64),
                    logits=g_outputs)))
        return {
            self.g: tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            self.d: tf.add_n(tf.get_collection('d_losses'), name='total_d_loss'),
        }

    def train_d(self, losses, learning_rate=0.0002, beta1=0.5):
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        d_opt_op = d_opt.minimize(losses[self.d], var_list=self.d.variables)
        return d_opt_op

    def train_g(self, losses, learning_rate=0.0002, beta1=0.5):
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        g_opt_op = g_opt.minimize(losses[self.g], var_list=self.g.variables)
        return g_opt_op

    def sample_images(self, row=8, col=8, inputs=None):
        if inputs is None:
            inputs = self.z

        images = self.g(inputs, training=True)
        # invert the image back to [0, 1]
        images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8)
        images = [image for image in tf.split(images, self.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
        image = tf.concat(rows, 1)
        return tf.image.encode_jpeg(tf.squeeze(image, [0]))


class Encoder:
    def __init__(self, depths=[64, 128, 256, 512], channel=3, z_dim=100, vae=True):
        self.depths = [channel] + depths
        self.reuse = False
        self.z_dim = z_dim
        self.vae = vae

    def __call__(self, inputs, training=False):
        outputs_layer = []

        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)

        outputs = tf.convert_to_tensor(inputs)
        outputs_layer.append(outputs)

        with tf.variable_scope('encode', reuse=self.reuse):
            # convolution x 4
            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                outputs_layer.append(outputs)
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                outputs_layer.append(outputs)
            with tf.variable_scope('conv3'):
                outputs = tf.layers.conv2d(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                outputs_layer.append(outputs)
            with tf.variable_scope('conv4'):
                outputs = tf.layers.conv2d(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                outputs_layer.append(outputs)
            with tf.variable_scope('latent'):
                batch_size = outputs.get_shape()[0].value
                reshape = tf.reshape(outputs, [batch_size, -1])
                outputs_layer.append(reshape)
                self.z_mu = tf.layers.dense(reshape, self.z_dim, name='outputs_z_mu')
                if self.vae:
                    self.z_log_sigma_sq = tf.layers.dense(reshape, self.z_dim, name='outputs_z_log_sigma_sq')
                    eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
                    outputs = self.z_mu + tf.exp(0.5 * self.z_log_sigma_sq) * eps
                    outputs_layer.append((self.z_mu, self.z_log_sigma_sq))
                else:
                    outputs = self.z_mu
                    outputs_layer.append(self.z_mu)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encode')
        self.outputs_layer = outputs_layer
        return outputs


class Decoder:
    def __init__(self, depths=[1024, 512, 256, 128], channel=3, s_size=4):
        self.depths = depths + [channel]
        self.s_size = s_size
        self.reuse = False

    def __call__(self, inputs, training=False):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('decode', reuse=self.reuse):
            # reshape from inputs
            with tf.variable_scope('reshape'):
                outputs = tf.layers.dense(inputs, self.depths[0] * self.s_size * self.s_size)
                outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            # deconvolution (transpose of convolution) x 4
            with tf.variable_scope('deconv1'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv2'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv3'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv4'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
            # output images
            with tf.variable_scope('tanh'):
                outputs = tf.tanh(outputs, name='outputs')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decode')
        return outputs


class VAE:
    def __init__(self,
                 batch_size=128, s_size=4, z_dim=100,
                 dec_depths=[1024, 512, 256, 128],
                 enc_depths=[64, 128, 256, 512],
                 channel=3,
                 vaemodel=False,
                 latent_weight=10 ** -5):
        self.batch_size = batch_size
        self.s_size = s_size
        self.z_dim = z_dim
        self.channel = channel
        self.vaemodel = vaemodel
        self.latent_weight = latent_weight
        self.encode = Encoder(channel=self.channel, z_dim=self.z_dim, vae=self.vaemodel)
        self.decode = Decoder(s_size=self.s_size, channel=self.channel)

    def loss(self, traindata):
        epsilon = 1e-10
        latents = self.encode(traindata, training=True)
        imgs_hat = self.decode(latents, training=True)
        recon_loss = tf.losses.mean_squared_error(traindata, imgs_hat)

        if self.vaemodel:
            latent_loss = -0.5 * tf.reduce_sum(
                1 + self.encode.z_log_sigma_sq - tf.square(self.encode.z_mu) - tf.exp(self.encode.z_log_sigma_sq),
                axis=1)
            latent_loss = tf.reduce_mean(latent_loss)
            vae_loss = recon_loss + latent_loss * self.latent_weight
            return {'vae_loss': vae_loss, 'latent_loss': latent_loss, 'recon_loss': recon_loss}
        else:
            return {'vae_loss': recon_loss, 'recon_loss': recon_loss}

    def train(self, losses, learning_rate=0.0002, beta1=0.5):
        vae_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        vae_opt_op = vae_opt.minimize(losses['vae_loss'])
        return vae_opt_op

    def decode_images(self, inputs=None):
        latents = self.encode(inputs, training=True)
        imgs_decode = self.decode(latents, training=True)
        return (latents, imgs_decode)


class VAEGAN(DCGAN):
    def __init__(self,
                 batch_size=128, s_size=4, z_dim=100,
                 dec_depths=[1024, 512, 256, 128],
                 enc_depths=[64, 128, 256, 512],
                 channel=3,
                 vaemodel=False,
                 latent_weight=10 ** -5):
        super().__init__(batch_size=batch_size, s_size=s_size, z_dim=z_dim, channel=channel)
        self.batch_size = batch_size
        self.s_size = s_size
        self.z_dim = z_dim
        self.channel = channel
        self.vaemodel = vaemodel
        self.latent_weight = latent_weight
        self.encode = Encoder(channel=self.channel, z_dim=self.z_dim, vae=self.vaemodel)
        self.decode = Decoder(s_size=self.s_size, channel=self.channel)
        self.g = self.decode

    def loss(self, traindata):
        epsilon = 1e-10
        latents = self.encode(traindata, training=True)
        imgs_hat = self.decode(latents, training=True)
        recon_loss = tf.losses.mean_squared_error(traindata, imgs_hat)

        self.z = latents
        ganlosses = super().loss(traindata)

        if self.vaemodel:
            latent_loss = -0.5 * tf.reduce_sum(
                1 + self.encode.z_log_sigma_sq - tf.square(self.encode.z_mu) - tf.exp(self.encode.z_log_sigma_sq),
                axis=1)
            latent_loss = tf.reduce_mean(latent_loss)
            vae_loss = recon_loss + latent_loss * self.latent_weight
            vaelosses = {'vae_loss': vae_loss, 'latent_loss': latent_loss, 'recon_loss': recon_loss}
        else:
            vaelosses = {'vae_loss': recon_loss, 'recon_loss': recon_loss}

        vaelosses.update(ganlosses)
        return vaelosses

    def vae_train(self, losses, learning_rate=0.0002, beta1=0.5):
        vae_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        vae_opt_op = vae_opt.minimize(losses['vae_loss'])
        return vae_opt_op

    def decode_images(self, inputs=None):
        latents = self.encode(inputs, training=True)
        imgs_decode = self.decode(latents, training=True)
        return (latents, imgs_decode)

