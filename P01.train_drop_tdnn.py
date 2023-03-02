import sys
import os
import json
import numpy as np
import random
import shutil
import math
import tensorflow as tf

from model_DC_ECAPA_TDNN_AMSoftmax import network

from absl import app
from absl import flags

from scipy import signal
from datetime import datetime


FLAGS = flags.FLAGS

flags.DEFINE_string('savemodel_dir', './exp/model', 'savemodel_dir.')
flags.DEFINE_string('savemodel_name', 'kaldi_dc_ecapa_tdnn_w_FC_w_RC_wo_PEL_amsoftmax_model_voxceleb2_train_mfcc80_2M_4gpus_batch128_clr', 'savemodel_name.')

flags.DEFINE_string('savedata_dir', './exp/data/voxceleb2_train_combined_no_sil/kaldi_features_mfcc80_All', 'savedata_dir.')
flags.DEFINE_string('phonedata_dir', './exp/model_phone/kaldi_resnet34_phone_model_librispeech_clean_100_mfcc80_batch256_clr_StopSteps/embeddings/voxceleb2_train_combined_no_sil-1300000', 'savedata_dir.')

flags.DEFINE_integer('epochs', 10, 'epochs.')
flags.DEFINE_integer('start_steps', 0, 'start_steps.')
flags.DEFINE_integer('stop_steps', 0, 'stop_steps.')
flags.DEFINE_integer('batch_size', 128, 'batch size.')
#flags.DEFINE_float('base_learning_rate', 1e-8, 'base_learning rate.')
#flags.DEFINE_float('max_learning_rate', 1e-3, 'max_learning rate.')
flags.DEFINE_float('base_learning_rate', 0.01, 'base_learning rate.')
flags.DEFINE_float('max_learning_rate', 0.1, 'max_learning rate.')

flags.DEFINE_integer('layer_num', 18, '18,34,50,101,152') # ResNet

flags.DEFINE_bool('is_multiple_embed', False, 'True or False.')
flags.DEFINE_bool('is_res_connect', False, 'True or False.')
flags.DEFINE_bool('is_fine_tuning', False, 'True or False.')

flags.DEFINE_bool('is_reconstruct_loss', True, 'True or False.')
flags.DEFINE_bool('is_frame_loss', True, 'True or False.')

flags.DEFINE_string('device_nums', '0,1,2,3', '0,1,2,3')
#os.environ['CUDA_VISIBLE_DEVICES'] = device_nums
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' #使用混合精度

layer_dim = 1024
pool_dim = 3 * layer_dim
embed_dim = 192
attention_channels=128
se_channels=128
res2net_scale_dim=8

para_m = 0.2
para_s = 30

embed_dim_p = 512

weight_decay_rate = 0.000002

num_steps_per_cycle = 130000
min_segment_len = 200
max_segment_len = 400

seg_num = 9 # the number of frames

lr_alpha = 1
lr_beta = 1
lr_gamma = 1
lr_delta = 0.01

data_num_perfile = 100000 # The amount of data saved per file.
num_steps_savemodel = 10000

######################
##### References #####
######################
# L2 Regularization
# >> https://stackoverflow.com/questions/44232566/add-l2-regularization-when-using-high-level-tf-layers/44238354
######################

def _train():

    global num_steps_per_cycle

    bgmodel_dir = os.path.join(FLAGS.savemodel_dir, FLAGS.savemodel_name)
    train_dir = bgmodel_dir+"/train_logs"

    if FLAGS.start_steps == 0:
        if os.path.exists(bgmodel_dir):
            shutil.rmtree(bgmodel_dir)
        #if not os.path.exists(bgmodel_dir):
        #    os.makedirs(bgmodel_dir)
        #if os.path.exists(train_dir):
        #    shutil.rmtree(train_dir)
        #if not os.path.exists(train_dir):
        #    os.makedirs(train_dir)

    device_list = FLAGS.device_nums.split(",")

    fp = open(os.path.join(FLAGS.savedata_dir)+"/data_size", "r")
    line = fp.readline()
    fp.close()

    data_size = int(line)

    print("\n")
    print("### data_size =", data_size)

    if data_size < FLAGS.batch_size:
        FLAGS.batch_size = data_size

    para_dict = {}

    fp = open(os.path.join(FLAGS.savedata_dir)+"/parameters.txt", "r")
    for line in iter(fp):
        temp_list = line.split("=")
        if len(temp_list) == 2:
            para_dict[temp_list[0]] = temp_list[1].replace("\n","")
    fp.close()

    num_samples_per_epoch = data_size
#    num_batches_per_epoch = num_steps_per_epoch #int(num_samples_per_epoch//FLAGS.batch_size)

    print("### data loading...")

    load_data_train = np.ndarray(num_samples_per_epoch, dtype=object)
    load_label_train = np.zeros((num_samples_per_epoch,), dtype=np.int32) #np.array([])
    load_pdata_train = np.ndarray(num_samples_per_epoch, dtype=object)
    load_plabel_train = np.zeros((num_samples_per_epoch,), dtype=np.int32) #np.array([])

    num_file_per_epoch = int(math.ceil(num_samples_per_epoch/data_num_perfile))

    print("\n")
    now = datetime.now()
    print(">>> date and time =", now.strftime("%d/%m/%Y %H:%M:%S"))

    for batch_file in range(num_file_per_epoch):

        load_data_train_sub = np.load(os.path.join(FLAGS.savedata_dir)+"/utterance_train_"+str(batch_file+1)+".npy", allow_pickle=True)
        load_label_train_sub = np.load(os.path.join(FLAGS.savedata_dir)+"/label_train_"+str(batch_file+1)+".npy")
        load_pdata_train_sub = np.load(os.path.join(FLAGS.phonedata_dir)+"/data_embeddings_"+str(batch_file+1)+".npy", allow_pickle=True)
        load_plabel_train_sub = np.load(os.path.join(FLAGS.phonedata_dir)+"/data_labels_"+str(batch_file+1)+".npy")

        # Convert to float16 data type
        for proc_i in range(load_data_train_sub.shape[0]):
            load_data_train_sub[proc_i] = load_data_train_sub[proc_i].astype(np.float16)
            load_pdata_train_sub[proc_i] = load_pdata_train_sub[proc_i].astype(np.float16)
  
        index_s = batch_file * data_num_perfile
        index_e = index_s + load_label_train_sub.shape[0]
  
        print(">>> index_s =", index_s, "index_e =", index_e, "load_label_train_sub.shape =", load_label_train_sub.shape)
  
        load_data_train[index_s:index_e] = load_data_train_sub
        load_label_train[index_s:index_e] = load_label_train_sub
        load_pdata_train[index_s:index_e] = load_pdata_train_sub
        load_plabel_train[index_s:index_e] = load_plabel_train_sub

    speaker_list = np.load(os.path.join(FLAGS.savedata_dir)+"/speaker_list.npy")

    num_subjects = speaker_list.shape[0]
    num_phones = load_pdata_train[0].shape[2]

    cut_num = seg_num // 2

    print("### num_subjects =", num_subjects)
    print("### num_phones =", num_phones)

    print("### model training...")

    use_device = "/cpu:0"
    if len(FLAGS.device_nums) > 0:
        use_device = "/gpu:"+FLAGS.device_nums.split(",")[0]

    with tf.Graph().as_default(), tf.device(use_device):

        # Create global_step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Training flag.
        is_training = tf.placeholder(tf.bool)

        # Define the place holders and creating the batch tensor.
        learning_rate = tf.placeholder(tf.float32, ())

        ### batch_speech >> [batch_size, utt_num, feature_dim]
        ### batch_labels >> [batch_size, spk_index]
        batch_speech = tf.placeholder(tf.float32, [None, None, int(para_dict["feature_dim"])], name="batch_speech")
        batch_labels = tf.placeholder(tf.int32, [None, 1], name="batch_labels")
        batch_pdata = tf.placeholder(tf.float32, [None, None, num_phones], name="batch_pdata")
        batch_plabels = tf.placeholder(tf.int32, [None, 1], name="batch_plabels")

        #opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        #opt = tf.train.GradientDescentOptimizer(learning_rate)

        # Distribute data among all clones equally.
        step = int(FLAGS.batch_size / float(len(device_list)))

        print("### model initializing...")
        model = network(num_subjects, num_phones, layer_dim, \
                pool_dim, embed_dim, attention_channels, se_channels, res2net_scale_dim, para_m, para_s, FLAGS.layer_num, embed_dim_p, seg_num)

        tower_grads = []
#        # [begin]-----------------------------------------------------------------
#        tvars = tf.trainable_variables()
#        # [end]-------------------------------------------------------------------

        with tf.variable_scope(tf.get_variable_scope()):

            total_loss_1 = tf.constant(0, dtype=tf.float16)
            total_loss_2 = tf.constant(0, dtype=tf.float16)
            total_loss = tf.constant(0, dtype=tf.float16)
            total_loss_p = tf.constant(0, dtype=tf.float16)
            total_loss_r = tf.constant(0, dtype=tf.float16)
            total_loss_frame = tf.constant(0, dtype=tf.float16)
            total_accuracy_1 = tf.constant(0, dtype=tf.float16)
            total_accuracy_2 = tf.constant(0, dtype=tf.float16)
            AM_total_accuracy_1 = tf.constant(0, dtype=tf.float16)
            AM_total_accuracy_2 = tf.constant(0, dtype=tf.float16)

            for gpu_idx, gpu_num in enumerate(device_list):
                with tf.device('/gpu:%s' % gpu_num):
                    with tf.name_scope('%s_%d' % ('tower', gpu_idx)) as scope:

                        ## Network inputs and outputs.
                        batch_input = batch_speech[gpu_idx * step: (gpu_idx + 1) * step]
                        batch_label = batch_labels[gpu_idx * step: (gpu_idx + 1) * step]
                        batch_pinput = batch_pdata[gpu_idx * step: (gpu_idx + 1) * step]
                        batch_plabel = batch_plabels[gpu_idx * step: (gpu_idx + 1) * step]
                        frame_features, features_1, embeddings_1, logits_1, AM_logits_1, features_2, embeddings_2, logits_2, AM_logits_2, \
                                features_p, logits_p, features_r  = \
                                model.dc_ecapa_tdnn_amsoftmax_model(batch_input, batch_label, batch_pinput, batch_plabel, \
                                is_training, FLAGS.is_multiple_embed, FLAGS.is_res_connect, FLAGS.is_reconstruct_loss)

                        ## one_hot labeling
                        label_onehot = tf.one_hot(tf.squeeze(batch_label, [1]), depth=num_subjects, axis=-1)

                        SOFTMAX_1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits_1, labels=label_onehot)
                        SOFTMAX_2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=label_onehot)
                        AM_SOFTMAX_1 = tf.nn.softmax_cross_entropy_with_logits(logits=AM_logits_1, labels=label_onehot)
                        AM_SOFTMAX_2 = tf.nn.softmax_cross_entropy_with_logits(logits=AM_logits_2, labels=label_onehot)

                        if FLAGS.is_reconstruct_loss == True:
                            plabels_flatten = tf.contrib.layers.flatten(batch_pinput[:,cut_num:-cut_num,:])
                            plogits_flatten = tf.contrib.layers.flatten(logits_p)
                            rlabels_flatten = tf.contrib.layers.flatten(batch_input[:,cut_num:-cut_num,:])
                            rlogits_flatten = tf.contrib.layers.flatten(features_r)

                            SIGMOID_p = tf.losses.mean_squared_error(plabels_flatten, plogits_flatten)
                            SIGMOID_r = tf.losses.mean_squared_error(rlabels_flatten, rlogits_flatten)

                        # Define loss
                        with tf.name_scope('loss'):
                            loss_1 = tf.reduce_mean(AM_SOFTMAX_1)
                            loss_2 = tf.reduce_mean(AM_SOFTMAX_2)
                            if FLAGS.is_multiple_embed == True:
                                loss = loss_1 + loss_2
                            else:
                                loss = loss_2

                            l2_loss = tf.losses.get_regularization_loss()

                            if FLAGS.is_frame_loss == True:
                                frame_features_mean = tf.expand_dims(tf.reduce_mean(frame_features, 1), 1)
                                loss_frame = tf.reduce_mean(tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.square(frame_features - frame_features_mean), 2)), 1))
                            else:
                                loss_frame = 0.0 #tf.reduce_mean(tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.square(frame_features - frame_features), 2)), 1))

                            if FLAGS.is_reconstruct_loss == True:
                                loss_p = tf.reduce_mean(SIGMOID_p)
                                loss_r = tf.reduce_mean(SIGMOID_r)
                                loss = lr_alpha * loss + lr_beta * loss_p + lr_gamma * loss_r + lr_delta * loss_frame + l2_loss
                            else:
                                loss_p = loss - loss
                                loss_r = loss - loss
                                loss = lr_alpha * loss + lr_delta * loss_frame + l2_loss

                        # Accuracy
                        with tf.name_scope('accuracy'):
                            # Evaluate the model
                            correct_pred_1 = tf.equal(tf.argmax(logits_1, 1), tf.argmax(label_onehot, 1))
                            correct_pred_2 = tf.equal(tf.argmax(logits_2, 1), tf.argmax(label_onehot, 1))
                            AM_correct_pred_1 = tf.equal(tf.argmax(AM_logits_1, 1), tf.argmax(label_onehot, 1))
                            AM_correct_pred_2 = tf.equal(tf.argmax(AM_logits_2, 1), tf.argmax(label_onehot, 1))

                            # Accuracy calculation
                            accuracy_1 = tf.reduce_mean(tf.cast(correct_pred_1, tf.float16))
                            accuracy_2 = tf.reduce_mean(tf.cast(correct_pred_2, tf.float16))
                            AM_accuracy_1 = tf.reduce_mean(tf.cast(AM_correct_pred_1, tf.float16))
                            AM_accuracy_2 = tf.reduce_mean(tf.cast(AM_correct_pred_2, tf.float16))

                        total_loss_1 = total_loss_1 + tf.cast(loss_1, tf.float16)
                        total_loss_2 = total_loss_2 + tf.cast(loss_2, tf.float16)
                        total_loss = total_loss + tf.cast(loss, tf.float16)
                        total_loss_p = total_loss_p + tf.cast(loss_p, tf.float16)
                        total_loss_r = total_loss_r + tf.cast(loss_r, tf.float16)
                        total_loss_frame = total_loss_frame + tf.cast(loss_frame, tf.float16)
                        total_accuracy_1 = total_accuracy_1 + tf.cast(accuracy_1, tf.float16)
                        total_accuracy_2 = total_accuracy_2 + tf.cast(accuracy_2, tf.float16)
                        AM_total_accuracy_1 = AM_total_accuracy_1 + tf.cast(AM_accuracy_1, tf.float16)
                        AM_total_accuracy_2 = AM_total_accuracy_2 + tf.cast(AM_accuracy_2, tf.float16)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)
#                        # [begin]-----------------------------------------------------------------
#                        grads = tf.gradients(loss, tvars)
#                        # [end]-------------------------------------------------------------------

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

            total_loss_1 = total_loss_1 / len(device_list)
            total_loss_2 = total_loss_2 / len(device_list)
            total_loss = total_loss / len(device_list)
            total_loss_p = total_loss_p / len(device_list)
            total_loss_r = total_loss_r / len(device_list)
            total_loss_frame = total_loss_frame / len(device_list)
            total_accuracy_1 = total_accuracy_1 / len(device_list)
            total_accuracy_2 = total_accuracy_2 / len(device_list)
            AM_total_accuracy_1 = AM_total_accuracy_1 / len(device_list)
            AM_total_accuracy_2 = AM_total_accuracy_2 / len(device_list)

#            tf.summary.scalar("loss_1", total_loss_1)
#            tf.summary.scalar("loss_2", total_loss_2)
#            tf.summary.scalar("loss", total_loss)
#            tf.summary.scalar("accuracy_1", total_accuracy_1)
#            tf.summary.scalar("accuracy_2", total_accuracy_2)
#            tf.summary.scalar("AM_accuracy_1", total_accuracy_1)
#            tf.summary.scalar("AM_accuracy_2", total_accuracy_2)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = model.average_gradients(tower_grads)
#        # [begin]-----------------------------------------------------------------
#        grads = model.average_gradients_only(tower_grads)
#        # [end]-------------------------------------------------------------------

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            # Apply the gradients to adjust the shared variables.
            train_op = opt.apply_gradients(grads, global_step=global_step)
#            # [begin]-----------------------------------------------------------------
#            optimizer = AdamWeightDecayOptimizer(
#                learning_rate=learning_rate,
#                weight_decay_rate=weight_decay_rate,
#                beta_1=0.9,
#                beta_2=0.999,
#                epsilon=1e-6)
#
#            # You can do clip gradients if you need in this step(in general it is not neccessary)
#            # (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
#            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
#            # [end]-------------------------------------------------------------------


        ###########################
        ######## Training #########
        ###########################

        # Initialization of the network.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)
        coord = tf.train.Coordinator()

        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

#            merged = tf.summary.merge_all()
#            writer = tf.summary.FileWriter(bgmodel_dir + "/TensorBoard/", graph = sess.graph)

#            save_record_count_all = 0
#            num_record_per_epoch = 100
#
#            save_index = math.ceil(num_batches_per_epoch / num_record_per_epoch)
#            num_record_per_epoch = int(num_batches_per_epoch / save_index)
            total_steps = FLAGS.epochs * num_steps_per_cycle

            # replace "FLAGS.epochs * num_record_per_epoch" with "total_steps"
            loss_1_epoch = np.zeros((total_steps, ), dtype=np.float32)
            loss_2_epoch = np.zeros((total_steps, ), dtype=np.float32)
            loss_epoch = np.zeros((total_steps, ), dtype=np.float32)
            loss_p_epoch = np.zeros((total_steps, ), dtype=np.float32)
            loss_r_epoch = np.zeros((total_steps, ), dtype=np.float32)
            loss_frame_epoch = np.zeros((total_steps, ), dtype=np.float32)
            acc_1_epoch = np.zeros((total_steps, ), dtype=np.float32)
            acc_2_epoch = np.zeros((total_steps, ), dtype=np.float32)
            AM_acc_1_epoch = np.zeros((total_steps, ), dtype=np.float32)
            AM_acc_2_epoch = np.zeros((total_steps, ), dtype=np.float32)
            lr_epoch = np.zeros((total_steps, ), dtype=np.float32)

            if FLAGS.start_steps > 0:
                # Restore model.
                #latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=bgmodel_dir)
                #saver.restore(sess, latest_checkpoint)
                checkpoint = os.path.join(bgmodel_dir, "train_logs-" + str(FLAGS.start_steps))
                saver.restore(sess, checkpoint)

                loss_1_epoch_tmp = np.load(os.path.join(bgmodel_dir)+"/loss_1_epoch.npy")
                loss_2_epoch_tmp = np.load(os.path.join(bgmodel_dir)+"/loss_2_epoch.npy")
                loss_epoch_tmp = np.load(os.path.join(bgmodel_dir)+"/loss_epoch.npy")
                loss_p_epoch_tmp = np.load(os.path.join(bgmodel_dir)+"/loss_p_epoch.npy")
                loss_r_epoch_tmp = np.load(os.path.join(bgmodel_dir)+"/loss_r_epoch.npy")
                loss_frame_epoch_tmp = np.load(os.path.join(bgmodel_dir)+"/loss_frame_epoch.npy")
                acc_1_epoch_tmp =  np.load(os.path.join(bgmodel_dir)+"/acc_1_epoch.npy")
                acc_2_epoch_tmp =  np.load(os.path.join(bgmodel_dir)+"/acc_2_epoch.npy")
                AM_acc_1_epoch_tmp =  np.load(os.path.join(bgmodel_dir)+"/am_acc_1_epoch.npy")
                AM_acc_2_epoch_tmp =  np.load(os.path.join(bgmodel_dir)+"/am_acc_2_epoch.npy")
                lr_epoch_tmp =  np.load(os.path.join(bgmodel_dir)+"/lr_epoch.npy")
                if loss_epoch_tmp.shape[0] < loss_epoch.shape[0]:
                    loss_1_epoch[:loss_1_epoch_tmp.shape[0]] = loss_1_epoch_tmp
                    loss_2_epoch[:loss_2_epoch_tmp.shape[0]] = loss_2_epoch_tmp
                    loss_epoch[:loss_epoch_tmp.shape[0]] = loss_epoch_tmp
                    loss_p_epoch[:loss_p_epoch_tmp.shape[0]] = loss_p_epoch_tmp
                    loss_r_epoch[:loss_r_epoch_tmp.shape[0]] = loss_r_epoch_tmp
                    loss_frame_epoch[:loss_frame_epoch_tmp.shape[0]] = loss_frame_epoch_tmp
                    acc_1_epoch[:acc_1_epoch_tmp.shape[0]] = acc_1_epoch_tmp
                    acc_2_epoch[:acc_2_epoch_tmp.shape[0]] = acc_2_epoch_tmp
                    AM_acc_1_epoch[:AM_acc_1_epoch_tmp.shape[0]] = AM_acc_1_epoch_tmp
                    AM_acc_2_epoch[:AM_acc_2_epoch_tmp.shape[0]] = AM_acc_2_epoch_tmp
                    lr_epoch[:lr_epoch_tmp.shape[0]] = lr_epoch_tmp
                else:
                    loss_1_epoch = loss_1_epoch_tmp
                    loss_2_epoch = loss_2_epoch_tmp
                    loss_epoch = loss_epoch_tmp
                    loss_p_epoch = loss_p_epoch_tmp
                    loss_r_epoch = loss_r_epoch_tmp
                    loss_frame_epoch = loss_frame_epoch_tmp
                    acc_1_epoch = acc_1_epoch_tmp
                    acc_2_epoch = acc_2_epoch_tmp
                    AM_acc_1_epoch = AM_acc_1_epoch_tmp
                    AM_acc_2_epoch = AM_acc_2_epoch_tmp
                    lr_epoch = lr_epoch_tmp
#                save_record_count_all = FLAGS.start_epoch * num_record_per_epoch

            early_stopping_loss = 0.001
            early_stopping_count = 5
            early_stopping_loss_check = 0
            early_stopping_acc_check = 0
            early_stopping_no_impv = 0

            prev_loss = -999.9
            
            if FLAGS.is_fine_tuning:
                pre_epochs_org = math.ceil(FLAGS.start_steps / num_steps_per_cycle)
                p_step_org = pre_epochs_org * num_steps_per_cycle 
                decay_rate = 0.0
                pre_epochs = 0
                pre_steps = 0
                num_steps_per_cycle = 60000
            else:
                pre_epochs_org = 0
                p_step_org = 0
                decay_rate = 0.25
                pre_epochs = FLAGS.start_steps // num_steps_per_cycle
                pre_steps = FLAGS.start_steps % num_steps_per_cycle

            for decay_i in range(pre_epochs):
                FLAGS.max_learning_rate = FLAGS.max_learning_rate * (1 - decay_rate)
                FLAGS.base_learning_rate = FLAGS.base_learning_rate * (1 - decay_rate)

            lr = FLAGS.base_learning_rate
            num_steps_per_cycle_l = math.ceil(num_steps_per_cycle / 2)
            num_steps_per_cycle_r = num_steps_per_cycle - num_steps_per_cycle_l
            diff_lr_per_step_l = (FLAGS.max_learning_rate - FLAGS.base_learning_rate) / (num_steps_per_cycle_l - 1)
            diff_lr_per_step_r = (FLAGS.max_learning_rate - FLAGS.base_learning_rate) / (num_steps_per_cycle_r)

            for cyclical_step in range(pre_steps):
                if cyclical_step == 0:
                    lr = FLAGS.base_learning_rate
                else:
                    if cyclical_step < num_steps_per_cycle_l:
                        lr = lr + diff_lr_per_step_l
                    else:
                        lr = lr - diff_lr_per_step_r

            print(">>> init learning_rate =", lr)

            # shuffling
            index_tr = random.sample(range(load_data_train.shape[0]), load_data_train.shape[0])
            sel_index = 0

#            ## https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.signal.exponential.html
#            #tau = -(FLAGS.epochs-1) / np.log(0.1)
#            #win = signal.exponential(FLAGS.epochs, 0, tau, False)
#            #lr_list = win * lr
#
#            # https://www.itread01.com/content/1547231642.html
#            decay_rate = 0.1
#
#            for epoch in range(FLAGS.start_epoch):
#                if epoch % 10 == 0 and epoch >= 50:
#                    #lr = lr_list[epoch]
#                    #lr = FLAGS.learning_rate * decay_rate ** (epoch / (FLAGS.epochs-1))
#                    #lr = FLAGS.learning_rate * decay_rate ** ((epoch % FLAGS.cyclical_learning_rate_epochs) / (FLAGS.cyclical_learning_rate_epochs-1))
#                    lr = lr * (1 - decay_rate)
#
#            #print(">>> init learning_rate =", lr)

            step_count = FLAGS.start_steps

            for epoch in range(pre_epochs, FLAGS.epochs):

                training_count = 0

                traing_loss_max = 0
                traing_loss_min = 0
                traing_loss_all = 0
                traing_acc_mean = 0
                traing_acc_all = 0

                save_record_count = 0

#                if epoch % 10 == 0 and epoch >= 50:
#                    #lr = lr_list[epoch]
#                    #lr = FLAGS.learning_rate * decay_rate ** (epoch / (FLAGS.epochs-1))
#                    #lr = FLAGS.learning_rate * decay_rate ** ((epoch % FLAGS.cyclical_learning_rate_epochs) / (FLAGS.cyclical_learning_rate_epochs-1))
#                    lr = lr * (1 - decay_rate)

                # Loop over all batches
                for batch_num in range(pre_steps, num_steps_per_cycle):

                    cyclical_step = batch_num
                    pre_steps = 0

                    if cyclical_step == 0:
                        lr = FLAGS.base_learning_rate
                    else:
                        if cyclical_step < num_steps_per_cycle_l:
                            lr = lr + diff_lr_per_step_l
                        else:
                            lr = lr - diff_lr_per_step_r

                    if sel_index + FLAGS.batch_size > load_data_train.shape[0]:
                        # shuffling
                        index_tr = random.sample(range(load_data_train.shape[0]), load_data_train.shape[0])
                        sel_index = 0

                    segment_len = random.randint(min_segment_len, max_segment_len)
                    batch_speech_i = np.zeros((FLAGS.batch_size, segment_len, int(para_dict["feature_dim"])), dtype=np.float16)
                    batch_labels_i = np.zeros((FLAGS.batch_size, 1), dtype=np.int32)
                    batch_pdata_i = np.zeros((FLAGS.batch_size, segment_len, num_phones), dtype=np.float16)
                    batch_plabels_i = np.zeros((FLAGS.batch_size, 1), dtype=np.int32)

                    #sel_index = random.randint(0, load_data_train.shape[0]-1)
                    #sel_speech_i = load_data_train[sel_index]
                    # [begin]-----------------------------------------------------------------
                    sel_speech_i = load_data_train[index_tr[sel_index]]
                    sel_pdata_i = load_pdata_train[index_tr[sel_index]]
                    # [end]-------------------------------------------------------------------

                    if sel_speech_i.shape[1] < segment_len:
                        segment_len = sel_speech_i.shape[1]
                        batch_speech_i = batch_speech_i[:,:segment_len,:]
                        batch_speech_i[0] = sel_speech_i
                        batch_pdata_i = batch_pdata_i[:,:segment_len,:]
                        batch_pdata_i[0] = sel_pdata_i
                    else:
                        index_s = random.randint(0, sel_speech_i.shape[1] - segment_len)
                        batch_speech_i[0] = sel_speech_i[:,index_s:index_s+segment_len,:]
                        batch_pdata_i[0] = sel_pdata_i[:,index_s:index_s+segment_len,:]

                    #batch_labels_i[0] = load_label_train[sel_index]
                    # [begin]-----------------------------------------------------------------
                    batch_labels_i[0] = load_label_train[index_tr[sel_index]]
                    batch_plabels_i[0] = load_plabel_train[index_tr[sel_index]]
                    # [end]-------------------------------------------------------------------

                    # [begin]-----------------------------------------------------------------
                    sel_index = sel_index + 1
                    # [end]-------------------------------------------------------------------
                    sel_speaker_count = 1

                    for proc_i in range(1, FLAGS.batch_size):

                        while sel_speaker_count == proc_i:

                            #sel_index = random.randint(0, load_data_train.shape[0]-1)
                            #sel_labels_i = load_label_train[sel_index]
                            # [begin]-----------------------------------------------------------------
                            sel_labels_i = load_label_train[index_tr[sel_index]]
                            sel_plabels_i = load_plabel_train[index_tr[sel_index]]
                            # [end]-------------------------------------------------------------------

                            #if sel_labels_i not in batch_labels_i:
                            # [begin]-----------------------------------------------------------------
                            if 1 == 1:
                            # [end]-------------------------------------------------------------------

                                #sel_speech_i = load_data_train[sel_index]
                                # [begin]-----------------------------------------------------------------
                                sel_speech_i = load_data_train[index_tr[sel_index]]
                                sel_pdata_i = load_pdata_train[index_tr[sel_index]]
                                # [end]-------------------------------------------------------------------

                                if sel_speech_i.shape[1] < segment_len:
                                    segment_len = sel_speech_i.shape[1]
                                    batch_speech_i = batch_speech_i[:,:segment_len,:]
                                    batch_speech_i[proc_i] = sel_speech_i
                                    batch_pdata_i = batch_pdata_i[:,:segment_len,:]
                                    batch_pdata_i[proc_i] = sel_pdata_i
                                else:
                                    index_s = random.randint(0, sel_speech_i.shape[1] - segment_len)
                                    batch_speech_i[proc_i] = sel_speech_i[:,index_s:index_s+segment_len,:]
                                    batch_pdata_i[proc_i] = sel_pdata_i[:,index_s:index_s+segment_len,:]

                                batch_labels_i[proc_i] = sel_labels_i
                                batch_plabels_i[proc_i] = sel_plabels_i

                                # [begin]-----------------------------------------------------------------
                                sel_index = sel_index + 1
                                # [end]-------------------------------------------------------------------
                                sel_speaker_count = sel_speaker_count + 1

                    # [begin]-----------------------------------------------------------------
                    #_, loss_1_value, loss_2_value, loss_value, \
                    #        acc_1_value, acc_2_value, AM_acc_1_value, AM_acc_2_value, \
                    #        training_step, training_merged = \
                    #        sess.run([train_op, total_loss_1, total_loss_2, total_loss, \
                    #        total_accuracy_1, total_accuracy_2, AM_total_accuracy_1, AM_total_accuracy_2, \
                    #        global_step, merged], \
                    #        feed_dict={learning_rate: lr, is_training: True, \
                    #        batch_speech: batch_speech_i, batch_labels: batch_labels_i})
                    _, loss_1_value, loss_2_value, loss_value, loss_p_value, loss_r_value, loss_frame_value, \
                            acc_1_value, acc_2_value, AM_acc_1_value, AM_acc_2_value, \
                            training_step = \
                            sess.run([train_op, total_loss_1, total_loss_2, total_loss, total_loss_p, total_loss_r, total_loss_frame, \
                            total_accuracy_1, total_accuracy_2, AM_total_accuracy_1, AM_total_accuracy_2, \
                            global_step], \
                            feed_dict={learning_rate: lr, is_training: True, \
                            batch_speech: batch_speech_i, batch_labels: batch_labels_i, \
                            batch_pdata: batch_pdata_i, batch_plabels: batch_plabels_i})
                    # [end]-------------------------------------------------------------------
    
                    training_count = training_count + 1
    
                    # # log
                    if training_count % 1 == 0:

                        now = datetime.now()

                        print("[" + now.strftime("%d/%m/%Y %H:%M:%S") + "] Epoch " + str(epoch + pre_epochs_org + 1) +
                                " : Minibatch " + str(batch_num + 1) + " of %d " % num_steps_per_cycle +
                                ", batch_size = " + str(FLAGS.batch_size) + ", segment_len = " + str(segment_len) +
                                ", sel_index = " + str(sel_index) + ", lr = " + str(lr))
                        if FLAGS.is_multiple_embed == True:
                            print(">>> Loss = " + "{:.4f}".format(loss_value) +
                                    ", Loss_1 = " + "{:.4f}".format(loss_1_value) +
                                    ", Loss_2 = " + "{:.4f}".format(loss_2_value) +
                                    ", Loss_p = " + "{:.4f}".format(loss_p_value) +
                                    ", Loss_r = " + "{:.4f}".format(loss_r_value) +
                                    ", Loss_frame = " + "{:.4f}".format(loss_frame_value))
                            print(">>> Acc_1 = " + "{:.3f}".format(100 * acc_1_value) +
                                    ", Acc_2 = " + "{:.3f}".format(100 * acc_2_value) +
                                    ", AM_Acc_1 = " + "{:.3f}".format(100 * AM_acc_1_value) +
                                    ", AM_Acc_2 = " + "{:.3f}".format(100 * AM_acc_2_value))
                        else:
                            print(">>> Loss = " + "{:.4f}".format(loss_value) +
                                    ", Loss_2 = " + "{:.4f}".format(loss_2_value) +
                                    ", Loss_p = " + "{:.4f}".format(loss_p_value) +
                                    ", Loss_r = " + "{:.4f}".format(loss_r_value) +
                                    ", Loss_frame = " + "{:.4f}".format(loss_frame_value) +
                                    ", Acc = " + "{:.3f}".format(100 * acc_2_value) +
                                    ", AM_Acc = " + "{:.3f}".format(100 * AM_acc_2_value))

                    #traing_loss_all = traing_loss_all + loss_value
                    #traing_acc_all = traing_acc_all + acc_value
    
                    ##if batch_file == 0 and sel_index == 0:
                    #if batch_num == 0:
                    #    traing_loss_max = loss_value
                    #    traing_loss_min = loss_value
                    #else:
                    #    if loss_value > traing_loss_max:
                    #        traing_loss_max = loss_value
                    #    if loss_value < traing_loss_min:
                    #        traing_loss_min = loss_value
    
#                    if batch_num % save_index == 0 and save_record_count < num_record_per_epoch:
#
#                        writer.add_summary(training_merged, epoch * num_batches_per_epoch + batch_num)
#
#                        loss_1_epoch[save_record_count_all] = loss_1_value
#                        loss_2_epoch[save_record_count_all] = loss_2_value
#                        loss_epoch[save_record_count_all] = loss_value
#                        acc_epoch[save_record_count_all] = acc_value
#                        save_record_count = save_record_count + 1
#                        save_record_count_all = save_record_count_all + 1

                    #loss_1_epoch[epoch * num_steps_per_cycle + batch_num + p_step_org] = loss_1_value
                    #loss_2_epoch[epoch * num_steps_per_cycle + batch_num + p_step_org] = loss_2_value
                    #loss_epoch[epoch * num_steps_per_cycle + batch_num + p_step_org] = loss_value
                    #acc_1_epoch[epoch * num_steps_per_cycle + batch_num + p_step_org] = acc_1_value
                    #acc_2_epoch[epoch * num_steps_per_cycle + batch_num + p_step_org] = acc_2_value
                    #AM_acc_1_epoch[epoch * num_steps_per_cycle + batch_num + p_step_org] = AM_acc_1_value
                    #AM_acc_2_epoch[epoch * num_steps_per_cycle + batch_num + p_step_org] = AM_acc_2_value
                    #lr_epoch[epoch * num_steps_per_cycle + batch_num + p_step_org] = lr
                    loss_1_epoch[step_count] = loss_1_value
                    loss_2_epoch[step_count] = loss_2_value
                    loss_epoch[step_count] = loss_value
                    loss_p_epoch[step_count] = loss_p_value
                    loss_r_epoch[step_count] = loss_r_value
                    loss_frame_epoch[step_count] = loss_frame_value
                    acc_1_epoch[step_count] = acc_1_value
                    acc_2_epoch[step_count] = acc_2_value
                    AM_acc_1_epoch[step_count] = AM_acc_1_value
                    AM_acc_2_epoch[step_count] = AM_acc_2_value
                    lr_epoch[step_count] = lr

                    #p_step = epoch * num_steps_per_cycle + batch_num + p_step_org + 1
                    step_count += 1

                    #if p_step % num_steps_savemodel == 0 or (FLAGS.stop_steps > 0 and FLAGS.stop_steps == p_step):
                    if step_count % num_steps_savemodel == 0 or (FLAGS.stop_steps > 0 and FLAGS.stop_steps == step_count):
                        # Save the model
                        #saver.save(sess, train_dir, global_step=p_step)
                        saver.save(sess, train_dir, global_step=step_count)

                        shutil.copy(os.path.join(FLAGS.savedata_dir)+"/speaker_count", os.path.join(bgmodel_dir)+"/speaker_count")

                        if FLAGS.is_fine_tuning:
                            filepath = bgmodel_dir+"/parameters_LM-FT-"+str(FLAGS.start_steps)+".txt"

                            if os.path.exists(filepath):
                                os.remove(filepath)

                            fp = open(filepath,"a")
                            fp.write("base_learning_rate="+str(FLAGS.base_learning_rate)+"\n")
                            fp.write("max_learning_rate="+str(FLAGS.max_learning_rate)+"\n")
                            fp.write("layer_num="+str(FLAGS.layer_num)+"\n")
                            fp.write("para_m="+str(para_m)+"\n")
                            fp.write("embed_dim_p="+str(embed_dim_p)+"\n")
                            fp.write("seg_num="+str(seg_num)+"\n")
                            fp.write("lr_alpha="+str(lr_alpha)+"\n")
                            fp.write("lr_beta="+str(lr_beta)+"\n")
                            fp.write("lr_gamma="+str(lr_gamma)+"\n")
                            fp.write("num_steps_per_cycle="+str(num_steps_per_cycle)+"\n")
                            fp.write("min_segment_len="+str(min_segment_len)+"\n")
                            fp.write("max_segment_len="+str(max_segment_len)+"\n")
                            fp.write("num_subjects="+str(num_subjects)+"\n")
                            fp.write("num_phones="+str(num_phones)+"\n")
                            fp.close()
                        else:
                            filepath = bgmodel_dir+"/parameters.txt"

                            if os.path.exists(filepath):
                                os.remove(filepath)

                            fp = open(filepath,"a")
                            fp.write("savedata_dir="+FLAGS.savedata_dir+"\n")
                            fp.write("epochs="+str(FLAGS.epochs)+"\n")
                            fp.write("batch_size="+str(FLAGS.batch_size)+"\n")
                            fp.write("base_learning_rate="+str(FLAGS.base_learning_rate)+"\n")
                            fp.write("max_learning_rate="+str(FLAGS.max_learning_rate)+"\n")
                            fp.write("layer_num="+str(FLAGS.layer_num)+"\n")
                            fp.write("feature_dim="+str(int(para_dict["feature_dim"]))+"\n")
                            fp.write("layer_dim="+str(layer_dim )+"\n")
                            fp.write("pool_dim="+str(pool_dim)+"\n")
                            fp.write("embed_dim="+str(embed_dim)+"\n")
                            fp.write("attention_channels="+str(attention_channels)+"\n")
                            fp.write("se_channels="+str(se_channels)+"\n")
                            fp.write("res2net_scale_dim="+str(res2net_scale_dim)+"\n")
                            fp.write("para_m="+str(para_m)+"\n")
                            fp.write("para_s="+str(para_s)+"\n")
                            fp.write("embed_dim_p="+str(embed_dim_p)+"\n")
                            fp.write("seg_num="+str(seg_num)+"\n")
                            fp.write("lr_alpha="+str(lr_alpha)+"\n")
                            fp.write("lr_beta="+str(lr_beta)+"\n")
                            fp.write("lr_gamma="+str(lr_gamma)+"\n")
                            fp.write("weight_decay_rate="+str(weight_decay_rate)+"\n")
                            fp.write("num_steps_per_cycle="+str(num_steps_per_cycle)+"\n")
                            fp.write("min_segment_len="+str(min_segment_len)+"\n")
                            fp.write("max_segment_len="+str(max_segment_len)+"\n")
                            fp.write("num_subjects="+str(num_subjects)+"\n")
                            fp.write("num_phones="+str(num_phones)+"\n")
                            fp.write("is_multiple_embed="+str(FLAGS.is_multiple_embed)+"\n")
                            fp.write("is_res_connect="+str(FLAGS.is_res_connect)+"\n")
                            fp.close()

                            np.save(os.path.join(bgmodel_dir)+"/loss_1_epoch.npy", loss_1_epoch)
                            np.save(os.path.join(bgmodel_dir)+"/loss_2_epoch.npy", loss_2_epoch)
                            np.save(os.path.join(bgmodel_dir)+"/loss_epoch.npy", loss_epoch)
                            np.save(os.path.join(bgmodel_dir)+"/loss_p_epoch.npy", loss_p_epoch)
                            np.save(os.path.join(bgmodel_dir)+"/loss_r_epoch.npy", loss_r_epoch)
                            np.save(os.path.join(bgmodel_dir)+"/loss_frame_epoch.npy", loss_frame_epoch)
                            np.save(os.path.join(bgmodel_dir)+"/acc_1_epoch.npy", acc_1_epoch)
                            np.save(os.path.join(bgmodel_dir)+"/acc_2_epoch.npy", acc_2_epoch)
                            np.save(os.path.join(bgmodel_dir)+"/am_acc_1_epoch.npy", AM_acc_1_epoch)
                            np.save(os.path.join(bgmodel_dir)+"/am_acc_2_epoch.npy", AM_acc_2_epoch)
                            np.save(os.path.join(bgmodel_dir)+"/lr_epoch.npy", lr_epoch)

                        #if FLAGS.stop_steps > 0 and FLAGS.stop_steps == p_step:
                        if FLAGS.stop_steps > 0 and FLAGS.stop_steps == step_count:
                            print("### Stop training at steps =", FLAGS.stop_steps)
                            exit()

                FLAGS.max_learning_rate = FLAGS.max_learning_rate * (1 - decay_rate)
                FLAGS.base_learning_rate = FLAGS.base_learning_rate * (1 - decay_rate)
                lr = FLAGS.base_learning_rate
                print(">>> init learning_rate =", lr)

#                if os.path.exists(bgmodel_dir+"/total_epoch"):
#                    os.remove(bgmodel_dir+"/total_epoch")
#
#                fp = open(bgmodel_dir+"/total_epoch","a")
#                fp.write(str(epoch+1))
#                fp.close()

                #if (traing_loss_max-traing_loss_min) <= early_stopping_loss:
                #    early_stopping_loss_check = early_stopping_loss_check + 1
                #else:
                #    early_stopping_loss_check = 0

                #if traing_acc_all/training_count <= traing_acc_mean:
                #    early_stopping_acc_check = early_stopping_acc_check + 1
                #else:
                #    early_stopping_acc_check = 0
                #    traing_acc_mean = traing_acc_all/training_count


                #if early_stopping_loss_check == early_stopping_count:
                #    print("### the training is stoped by the \"early_stopping_loss_check\"")
                #    break
                #if early_stopping_acc_check == early_stopping_count:
                #    print("### the training is stoped by the \"early_stopping_acc_check\"")
                #    break
                #if traing_loss_all/training_count < early_stopping_loss:
                #    print("### the training is stoped by the \"early_stopping_loss\"")
                #    break

                #if traing_loss_all/training_count > prev_loss:
                #    early_stopping_no_impv += 1
                #    #if early_stopping_no_impv >= 3:
                #    #    lr = lr / 2.0
                #    if early_stopping_no_impv >= 3:
                #        print("### No imporvement for 3 epochs, early stopping.")
                #        break
                #else:
                #    early_stopping_no_impv = 0

                #prev_loss = traing_loss_all/training_count


def main(argv):

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_nums

    if FLAGS.is_fine_tuning:
        global para_m, num_steps_per_cycle, min_segment_len, max_segment_len
        FLAGS.base_learning_rate = 1e-8
        FLAGS.max_learning_rate = 1e-5
        para_m = 0.5
        min_segment_len = 600
        max_segment_len = 600

    _train()

    print("### the DROP-TDNN AM-Softmax model was created.")


if __name__ == "__main__":
    app.run(main)
