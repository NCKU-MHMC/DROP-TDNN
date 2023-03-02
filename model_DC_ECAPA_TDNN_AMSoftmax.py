import math
import numpy as np
import tensorflow as tf

class network(object):
    def __init__(self, spk_num, phone_num, layer_size=1024, pooling_size=3072, embedding_size=192, \
            attention_channels=128, se_channels=128, res2net_scale_dim=8, \
            para_m=0.2, para_s=30, layer_num=18, embedding_size_p=512, seg_num=9):

        self.spk_num = spk_num
        self.layer_size = layer_size
        self.pooling_size = pooling_size
        self.embedding_size = embedding_size

        self.attention_channels = attention_channels
        self.se_channels = se_channels
        self.res2net_scale_dim = res2net_scale_dim

        self.para_m = para_m
        self.para_s = para_s

        self.phone_num = phone_num
        self.layer_num = layer_num
        self.embedding_size_p = embedding_size_p
        self.seg_num = seg_num  

        self.epsilon = 1e-9


    def LeakyReLU(self, inputs, leak=0.3, scope="LeakyReLU"):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)

        return f1 * inputs + f2 * tf.abs(inputs)


    # reference: https://github.com/tensorflow/tensorflow/issues/1122#issuecomment-280325584
    def bn_layer(self, x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
        """
        Performs a batch normalization layer

        Args:
            x: input tensor
            scope: scope name
            is_training: python boolean value
            epsilon: the variance epsilon - a small float number to avoid dividing by 0
            decay: the moving average decay

        Returns:
            The ops of a batch normalization layer
        """
        with tf.variable_scope(scope, reuse=reuse):
            shape = x.get_shape().as_list()
            # gamma: a trainable scale factor
            gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
            # beta: a trainable shift value
            beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
            moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
            moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
            if is_training:
                # tf.nn.moments == Calculate the mean and the variance of the tensor x
                avg, var = tf.nn.moments(x, np.arange(len(shape)-1), keep_dims=True)
                avg=tf.reshape(avg, [avg.shape.as_list()[-1]])
                var=tf.reshape(var, [var.shape.as_list()[-1]])
                #update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
                update_moving_avg=tf.assign(moving_avg, moving_avg*decay+avg*(1-decay))
                #update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
                update_moving_var=tf.assign(moving_var, moving_var*decay+var*(1-decay))
                control_inputs = [update_moving_avg, update_moving_var]
            else:
                avg = moving_avg
                var = moving_var
                control_inputs = []
            with tf.control_dependencies(control_inputs):
                output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

        return output


    def bn_layer_top(self, x, scope, is_training, epsilon=0.001, decay=0.99):
        """
        Returns a batch normalization layer that automatically switch between train and test phases based on the
        tensor is_training

        Args:
            x: input tensor
            scope: scope name
            is_training: boolean tensor or variable
            epsilon: epsilon parameter - see batch_norm_layer
            decay: epsilon parameter - see batch_norm_layer

        Returns:
            The correct batch normalization layer based on the value of is_training
        """
        #assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

        return tf.cond(
            is_training,
            lambda: self.bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=tf.AUTO_REUSE),
            lambda: self.bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
        )


    def AM_logits_compute(self, embeddings, labels, name):

        '''
        loss head proposed in paper:<Additive Margin Softmax for Face Verification>
        link: https://arxiv.org/abs/1801.05599

        embeddings : normalized embedding layer of Facenet, it's normalized value of output of resface
        labels : ground truth label of current training batch
        '''
        with tf.name_scope("AM_logits_{}".format(name)):
            kernel = tf.get_variable(name="kernel_{}".format(name), dtype=tf.float32, shape=[self.embedding_size, self.spk_num], \
                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            kernel_norm = tf.nn.l2_normalize(kernel, dim=0, epsilon=self.epsilon, name="kernel_norm_{}".format(name))
            cos_theta = tf.matmul(embeddings, kernel_norm)
            cos_theta = tf.clip_by_value(cos_theta, -1,1) # for numerical steady
            phi = cos_theta - self.para_m
            label_onehot = tf.one_hot(tf.squeeze(labels, [1]), depth=self.spk_num, axis=-1)
            adjust_theta = self.para_s * tf.where(tf.equal(label_onehot,1), phi, cos_theta)

            return cos_theta, adjust_theta


    def Res2Block(self, inputs, kernel_size, dilation_rate, scope, is_training):

        channel_dims = inputs.get_shape()[-1]

        partial_dims = channel_dims // self.res2net_scale_dim

        outputs = inputs[:, :, :partial_dims]
        
        for proc_i in range(1, self.res2net_scale_dim):
            partial_inputs = inputs[:, :, partial_dims*proc_i:partial_dims*(proc_i+1)]
            if proc_i > 1:
                partial_inputs = partial_inputs + net

            net = tf.layers.conv1d(partial_inputs, partial_inputs.get_shape()[-1], kernel_size, dilation_rate=dilation_rate, padding="same", \
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                    name="{}_res2_tdnn{}".format(scope, str(proc_i)))
            #net = tf.nn.relu(net)
            net = self.LeakyReLU(net)
            #net = tf.layers.batch_normalization(net, training=is_training, name="{}_res2_tdnn{}_bn".format(scope, str(proc_i)))
            net = self.bn_layer_top(net, "{}_res2_tdnn{}_bn".format(scope, str(proc_i)), is_training)

            outputs = tf.concat([outputs, net], 2)

        return outputs


    def SEBlock(self, inputs, scope, is_training):

        ### net >> [batch_size, frames, layer_size]
#        net = tf.reduce_mean(inputs, axis=1, keepdims=True)
        net = tf.layers.conv1d(inputs, self.se_channels, 1, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="{}_se_tdnn1".format(scope))
        #net = tf.nn.relu(net)
        net = self.LeakyReLU(net)
        #net = tf.layers.batch_normalization(net, training=is_training, name="{}_net_tdnn1_bn".format(scope))
        net = self.bn_layer_top(net, "{}_se_tdnn1_bn".format(scope), is_training)

        net = tf.layers.conv1d(net, self.layer_size, 1, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="{}_se_tdnn2".format(scope))
#        #net = tf.sigmoid(net)
#        net = tf.nn.softmax(net, axis=2)
#        outputs = tf.multiply(inputs, net) #inputs * net

        #net = tf.nn.relu(net)
        net = self.LeakyReLU(net)
        #net = tf.layers.batch_normalization(net, training=is_training, name="{}_net_tdnn1_bn".format(scope))
        net = self.bn_layer_top(net, "{}_se_tdnn2_bn".format(scope), is_training)
        outputs = net

        return outputs


    def SE_Res2Block(self, inputs, kernel_size, dilation_rate, is_residual, scope, is_training):

        ### net >> [batch_size, frames, layer_size]
        net = tf.layers.conv1d(inputs, self.layer_size, 1, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="{}_net_tdnn1".format(scope))
        #net = tf.nn.relu(net)
        net = self.LeakyReLU(net)
        #net = tf.layers.batch_normalization(net, training=is_training, name="{}_net_tdnn1_bn".format(scope))
        net = self.bn_layer_top(net, "{}_net_tdnn1_bn".format(scope), is_training)

        net = self.Res2Block(net, kernel_size, dilation_rate, scope, is_training)

        net = tf.layers.conv1d(net, self.layer_size, 1, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="{}_net_tdnn2".format(scope))
        #net = tf.nn.relu(net)
        net = self.LeakyReLU(net)
        #net = tf.layers.batch_normalization(net, training=is_training, name="{}_net_tdnn2_bn".format(scope))
        net = self.bn_layer_top(net, "{}_net_tdnn2_bn".format(scope), is_training)

        net = self.SEBlock(net, scope, is_training)

        if is_residual:
#            kernel_range = (kernel_size - 1) * dilation_rate + 1
#            left_size = kernel_range // 2
#            right_size = kernel_range - left_size - 1
#            outputs = inputs[:, left_size:-right_size, :] + net
            outputs = inputs + net

            return outputs
        else:
            return net


    def ecapa_tdnn_framelevel(self, inputs, is_training):

        # The initial TDNN layer
        net = tf.layers.conv1d(inputs, self.layer_size, 5, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="layer1")
        #net = tf.nn.relu(net)
        net = self.LeakyReLU(net)
        #net = tf.layers.batch_normalization(net, training=is_training, name="layer1_bn")
        net = self.bn_layer_top(net, "layer1_bn", is_training)
    
        # SE-Res2Net layers
        net_b1 = self.SE_Res2Block(net, 3, 2, True, "layer3", is_training)
    
        net_b2 = self.SE_Res2Block(net_b1, 3, 3, True, "layer5", is_training)
    
        net_b3 = self.SE_Res2Block(net_b2, 3, 4, True, "layer7", is_training)
        
        # Multi-layer feature aggregation
        net = tf.concat([net_b1, net_b2, net_b3], 2)
        net = tf.layers.conv1d(net, self.pooling_size, 1, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="layer9")
        #net = tf.nn.relu(net)
        net = self.LeakyReLU(net)
        #net = tf.layers.batch_normalization(net, training=is_training, name="layer9_bn")
        net = self.bn_layer_top(net, "layer9_bn", is_training)
    
        return net


    def attentive_statistics_pooling(self, inputs, is_training):

        net = tf.layers.conv1d(inputs, self.attention_channels, 1, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="attn_a")
#        #net = tf.nn.relu(net)
#        net = self.LeakyReLU(net)
#        #net = tf.layers.batch_normalization(net, training=is_training, name="attn_a_bn")
#        net = self.bn_layer_top(net, "attn_a_bn", is_training)
        net = tf.tanh(net)
        net = tf.layers.conv1d(net, self.pooling_size, 1, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="attn_b")

        attn = tf.transpose(net, perm=[0, 2, 1])
        attn = tf.nn.softmax(attn, axis=2)
        attn = tf.transpose(attn, perm=[0, 2, 1])

        mean = tf.reduce_sum(attn * inputs, axis=1, keepdims=True)
        #__X__std = tf.sqrt(tf.reduce_sum(attn * (inputs ** 2) - (mean ** 2), axis=1))
        std = tf.sqrt(tf.reduce_sum(attn * ((inputs - mean) ** 2), axis=1))
        mean = tf.squeeze(mean, [1])

        net = tf.concat([mean, std], 1)

        return net

    ###############################################################
    # PHONE SUB-MODEL
    ###############################################################
    def resnet_block1(self, net, channel_num, is_residual, is_training, scope):

        net_org = net

        net = tf.layers.conv2d(net, channel_num, (1, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="{}_layer1".format(scope))
        #net = tf.nn.relu(net)
        net = self.LeakyReLU(net)
        net = self.bn_layer_top(net, "{}_layer1_bn".format(scope), is_training)

        net = tf.layers.conv2d(net, channel_num, (1, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="{}_layer2".format(scope))
        #net = tf.nn.relu(net)
        net = self.LeakyReLU(net)
        net = self.bn_layer_top(net, "{}_layer2_bn".format(scope), is_training)

        if is_residual:
            net = net + net_org

        return net


    def resnet(self, inputs, is_training):

        block_layers = [2, 2, 2, 2] # default: 18 layers
        if self.layer_num == 34:
            block_layers = [3, 4, 6, 3]
        if self.layer_num == 50:
            block_layers = [3, 4, 6, 3]
        if self.layer_num == 101:
            block_layers = [3, 4, 23, 3]
        if self.layer_num == 152:
            block_layers = [3, 8, 36, 3]

        ### [batch_size, frames, feat_dim] ###

        ### [?, 9, 80] >> [?, 9, 80, 1] ###
        net = tf.expand_dims(inputs, axis=3)

        ### [?, 9, 80, 1] >> [?, 3, 37, 64] ###
        net = tf.layers.conv2d(net, 64, (7, 7), strides=(1, 2), padding='valid', dilation_rate=(1, 1), \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="layer1_p")
        #net = tf.nn.relu(net)
        net = self.LeakyReLU(net)
        net = self.bn_layer_top(net, "layer1_p_bn", is_training)

        ### [?, 3, 37, 64] >> [?, 1, 18, 64] ###
        net = tf.layers.max_pooling2d(net, (3, 3), strides=(1, 2))

        block_count = 0
        if self.layer_num == 18 or self.layer_num == 34:

            ### [?, 1, 18, 64] >> [?, 1, 18, 64] ###
            for proc_i in range(block_layers[0]):
                net = self.resnet_block1(net, 64, True, is_training, "block"+str(block_count+1)+"_p")
                block_count += 1

            ### [?, 1, 18, 64] >> [?, 1, 18, 128] ###
            for proc_i in range(block_layers[1]):
                net = self.resnet_block1(net, 128, True if proc_i > 0 else False, is_training, "block"+str(block_count+1)+"_p")
                block_count += 1

            ### [?, 1, 18, 128] >> [?, 1, 18, 256] ###
            for proc_i in range(block_layers[2]):
                net = self.resnet_block1(net, 256, True if proc_i > 0 else False, is_training, "block"+str(block_count+1)+"_p")
                block_count += 1

            ### [?, 1, 18, 256] >> [?, 1, 18, embedding_size_p] ###
            for proc_i in range(block_layers[3]):
                net = self.resnet_block1(net, self.embedding_size_p, True if proc_i > 0 else False, is_training, "block"+str(block_count+1)+"_p")
                block_count += 1

            ### [?, 1, 18, embedding_size_p] >> [?, 1, embedding_size_p] ###
            net = tf.layers.average_pooling2d(net, (1, 18), (1, 1))
            net = tf.squeeze(net, [2])

        else:

            bbb = 0


        return net


    def phone_resnet_model(self, inputs, is_training):

        net = self.resnet(inputs, is_training)

        #embeddings = net
        #embeddings = tf.nn.l2_normalize(embeddings, dim=1, epsilon=self.epsilon, name='embeddings_p')
        #logits, AM_logits = self.AM_logits_compute(embeddings, labels, "type_p")

        #logits = tf.identity(logits, name='outputs_p')
        #AM_logits = tf.identity(AM_logits, name='AM_outputs_p')

        logits = tf.layers.conv1d(net, self.phone_num, 1, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="layer_p")

        return net, logits


    def phone_ann_model(self, inputs, is_training):

        net = tf.layers.conv1d(inputs, self.embedding_size_p, self.seg_num, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="layer1_p")
        #net = tf.nn.relu(net)
        net = self.LeakyReLU(net)
        #net = tf.layers.batch_normalization(net, training=is_training, name="layer1_p_bn")
        net = self.bn_layer_top(net, "layer1_p_bn", is_training)

        for proc_i in range(1, 5):
            net = tf.layers.conv1d(net, self.embedding_size_p, 1, dilation_rate=1, \
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                    name="layer{}_p".format(str(proc_i+1)))
            #net = tf.nn.relu(net)
            net = self.LeakyReLU(net)
            #net = tf.layers.batch_normalization(net, training=is_training, name="layer{}_p_bn".format(str(proc_i+1)))
            net = self.bn_layer_top(net, "layer{}_p_bn".format(str(proc_i+1)), is_training)

        logits = tf.layers.conv1d(net, self.phone_num, 1, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="layer_output_p")

        return net, logits


    ###############################################################
    # RECONSTRUCT MODEL
    ###############################################################
    def reconstruct_model(self, inputs, ref_outputs, is_training):

        net = tf.layers.conv1d(inputs, self.layer_size, 1, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="layer1_r")
        #net = tf.nn.relu(net)
        net = self.LeakyReLU(net)
        #net = tf.layers.batch_normalization(net, training=is_training, name="layer1_bn")
        net = self.bn_layer_top(net, "layer1_r_bn", is_training)

        net = tf.layers.conv1d(net, self.layer_size, 1, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="layer2_r")
        #net = tf.nn.relu(net)
        net = self.LeakyReLU(net)
        #net = tf.layers.batch_normalization(net, training=is_training, name="layer2_bn")
        net = self.bn_layer_top(net, "layer2_r_bn", is_training)

        net = tf.layers.conv1d(net, ref_outputs.get_shape()[-1], 1, dilation_rate=1, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="layer3_r")

        return net


    def dc_ecapa_tdnn_amsoftmax_model(self, inputs, labels, pinputs, plabels, is_training, \
            is_multiple_embed=True, is_res_connect=False, is_reconstruct=True):

        net = self.ecapa_tdnn_framelevel(inputs, is_training)
        frame_features = net

        if is_reconstruct == True:
            ####################################################################################################
            ### Divide and Conquer Process
            #features_p, logits_p = self.phone_resnet_model(inputs, is_training)
            features_p, logits_p = self.phone_ann_model(inputs, is_training)

            net_r = tf.concat([net[:,2:-2,:], features_p], 2) # net[:,2:-2,:] -> the first kernel size of conv1d is 5

            features_r = self.reconstruct_model(net_r, inputs, is_training)
            ####################################################################################################
        else:
            ### Fake data
            features_p = pinputs
            logits_p = plabels
            features_r = inputs
      
        #net_mean, net_var = tf.nn.moments(net, axes=[1])
        #net = tf.identity(tf.concat([net_mean, tf.sqrt(net_var)], 1), name='layer10')
        net = self.attentive_statistics_pooling(net, is_training)

        features_1 = tf.layers.dense(net, self.embedding_size, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="layer11")
        embeddings_1 = features_1
        embeddings_1 = tf.nn.l2_normalize(embeddings_1, dim=1, epsilon=self.epsilon, name='embeddings_1')
        #net_1 = tf.nn.relu(features_1)
        net_1 = self.LeakyReLU(features_1)
        #net_1 = tf.layers.batch_normalization(net_1, training=is_training, name="layer11_bn")
        net_1 = self.bn_layer_top(net_1, "layer11_bn", is_training)

        if is_multiple_embed == True:

            #embeddings_1 = tf.layers.dropout(embeddings_1, rate=0.25, training=is_training)
            logits_1, AM_logits_1 = self.AM_logits_compute(embeddings_1, labels, "type_1")
 
#            scope.reuse_variables()

        features_2 = tf.layers.dense(net_1, self.embedding_size, \
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), \
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=2e-5), \
                name="layer12")

        if is_res_connect == True:
            features_2 = features_1 + features_2

        embeddings_2 = features_2
        embeddings_2 = tf.nn.l2_normalize(embeddings_2, dim=1, epsilon=self.epsilon, name='embeddings_2')
        #embeddings_2 = tf.layers.dropout(embeddings_2, rate=0.25, training=is_training)
        logits_2, AM_logits_2 = self.AM_logits_compute(embeddings_2, labels, "type_2")
      
        if is_multiple_embed == True:
            logits_1 = tf.identity(logits_1, name='outputs_1')
            logits_2 = tf.identity(logits_2, name='outputs_2')
            AM_logits_1 = tf.identity(AM_logits_1, name='AM_outputs_1')
            AM_logits_2 = tf.identity(AM_logits_2, name='AM_outputs_2')

            return frame_features, features_1, embeddings_1, logits_1, AM_logits_1, features_2, embeddings_2, logits_2, AM_logits_2, \
                    features_p, logits_p, features_r
        else:
            logits_2 = tf.identity(logits_2, name='outputs_2')
            AM_logits_2 = tf.identity(AM_logits_2, name='AM_outputs_2')

            return frame_features, features_1, embeddings_1, logits_2, AM_logits_2, features_2, embeddings_2, logits_2, AM_logits_2, \
                    features_p, logits_p, features_r


    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


    def average_gradients_only(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            average_grads.append(grad)
        return average_grads

