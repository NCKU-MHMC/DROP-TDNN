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
flags.DEFINE_string('saveembedding_name', 'voxceleb2_train_combined_no_sil', 'saveembedding_name.')

flags.DEFINE_integer('load_steps', 0, 'load_steps.')

flags.DEFINE_integer('start_file_index', 0, 'start_file_index.')

flags.DEFINE_integer('layer_num', 18, '18,34,50,101,152') # ResNet

flags.DEFINE_bool('is_multiple_embed', False, 'True or False.')
flags.DEFINE_bool('is_res_connect', False, 'True or False.')
flags.DEFINE_string('device_nums', '0', 'device_nums.')

flags.DEFINE_bool('is_reconstruct_loss', True, 'True or False.')

#os.environ['CUDA_VISIBLE_DEVICES'] = device_nums
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' #使用混合精度

data_num_perfile = 100000 # The amount of data saved per file.

data_num_frames_max = 60000 # 60000 frames = 10 seconds


def _test():

    bgmodel_dir = os.path.join(FLAGS.savemodel_dir, FLAGS.savemodel_name)
    savedata_dir = os.path.join(bgmodel_dir, "embeddings", FLAGS.saveembedding_name + "-" + str(FLAGS.load_steps))

    if FLAGS.start_file_index == 0:

        if os.path.exists(os.path.join(savedata_dir)):
            shutil.rmtree(os.path.join(savedata_dir))

        subfields = savedata_dir.split("/")

        temp_dir = subfields[0]
        for proc_index in range(1, len(subfields)):
            temp_dir = os.path.join(temp_dir, subfields[proc_index])
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

    device_list = FLAGS.device_nums.split(",")

    fp = open(os.path.join(FLAGS.savedata_dir)+"/data_size", "r")
    line = fp.readline()
    fp.close()

    data_size = int(line)

    print("\n")
    print("### data_size =", data_size)

    para_dict = {}

    fp = open(os.path.join(bgmodel_dir)+"/parameters.txt", "r")
    for line in iter(fp):
        temp_list = line.split("=")
        if len(temp_list) == 2:
            para_dict[temp_list[0]] = temp_list[1].replace("\n","")
    fp.close()

    num_samples_per_epoch = data_size
#    num_batches_per_epoch = int(num_samples_per_epoch//FLAGS.batch_size)

    fp = open(os.path.join(bgmodel_dir)+"/speaker_count", "r")
    line = fp.readline()
    fp.close()

    speaker_count = int(line)

    num_subjects = speaker_count

    print("### num_subjects =", num_subjects)
    print("### num_phones =", int(para_dict["num_phones"]))

    print("### model testing...")

    use_device = "/cpu:0"
    if len(FLAGS.device_nums) > 0:
        use_device = "/gpu:"+FLAGS.device_nums.split(",")[0]

    with tf.Graph().as_default(), tf.device(use_device):

        # Training flag.
        is_training = tf.placeholder(tf.bool)

        ### batch_speech >> [batch_size, utt_num, feature_dim]
        ### batch_labels >> [batch_size, spk_index]
        batch_speech = tf.placeholder(tf.float32, [None, None, int(para_dict["feature_dim"])], name="batch_speech")
        batch_labels = tf.placeholder(tf.int32, [None, 1], name="batch_labels")        
        batch_pdata = tf.placeholder(tf.float32, [None, None, int(para_dict["num_phones"])], name="batch_pdata")
        batch_plabels = tf.placeholder(tf.int32, [None, 1], name="batch_plabels")

        print("### model initializing...")
        model = network(num_subjects, int(para_dict["num_phones"]), int(para_dict["layer_dim"]), int(para_dict["pool_dim"]), int(para_dict["embed_dim"]), \
                int(para_dict["attention_channels"]), int(para_dict["se_channels"]), int(para_dict["res2net_scale_dim"]), \
                float(para_dict["para_m"]), int(para_dict["para_s"]), FLAGS.layer_num, int(para_dict["embed_dim_p"]), int(para_dict["seg_num"]))

        with tf.variable_scope(tf.get_variable_scope()):

            ## Network inputs and outputs.
            batch_input = batch_speech #batch_speech[gpu_idx * step: (gpu_idx + 1) * step]
            batch_label = batch_labels #batch_labels[gpu_idx * step: (gpu_idx + 1) * step]
            batch_pinput = batch_pdata #batch_pdata[gpu_idx * step: (gpu_idx + 1) * step]
            batch_plabel = batch_plabels #batch_plabels[gpu_idx * step: (gpu_idx + 1) * step]
            frame_features, features_1, embeddings_1, logits_1, AM_logits_1, features_2, embeddings_2, logits_2, AM_logits_2, \
                    features_p, logits_p, features_r = model.dc_ecapa_tdnn_amsoftmax_model(\
                    batch_input, batch_label, batch_pinput, batch_plabel, is_training, \
                    FLAGS.is_multiple_embed, FLAGS.is_res_connect, FLAGS.is_reconstruct_loss)

        ###########################
        ######## Testing #########
        ###########################

        # Initialization of the network.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
        coord = tf.train.Coordinator()

        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Restore model.
            #latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=bgmodel_dir)
            #saver.restore(sess, latest_checkpoint)
            checkpoint = os.path.join(bgmodel_dir, "train_logs-" + str(FLAGS.load_steps))
            saver.restore(sess, checkpoint)


            num_file_per_epoch = int(math.ceil(num_samples_per_epoch/data_num_perfile))

            for batch_file in range(FLAGS.start_file_index, num_file_per_epoch):

                if not os.path.isfile(os.path.join(savedata_dir)+"/data_labels_"+str(batch_file+1)+".npy"):

                    print("### (", batch_file+1, "/", num_file_per_epoch, ") data loading...")

                    load_data_sub = np.load(os.path.join(FLAGS.savedata_dir)+"/utterance_train_"+str(batch_file+1)+".npy", allow_pickle=True)
                    #load_label_sub = np.load(os.path.join(FLAGS.savedata_dir)+"/label_train_"+str(batch_file+1)+".npy")
      
                    feature_vector_1 = np.zeros((load_data_sub.shape[0], int(para_dict["embed_dim"])), dtype=np.float32)
                    feature_vector_2 = np.zeros((load_data_sub.shape[0], int(para_dict["embed_dim"])), dtype=np.float32)

                    num_batches_per_file = load_data_sub.shape[0]

                    for batch_num in range(num_batches_per_file):

                        print("### (", batch_num+1, "/", num_batches_per_file, ") extract embedding...", end="\r")

                        batch_speech_i = load_data_sub[batch_num].astype(np.float16)
                        batch_labels_i = np.zeros((1,1), dtype=np.int32) #load_label_sub[batch_num]
                        if batch_speech_i.shape[1] > data_num_frames_max:
                            batch_speech_i = batch_speech_i[:,:data_num_frames_max,:]

                        batch_pdata_i = np.zeros((batch_speech_i.shape[0],batch_speech_i.shape[1],int(para_dict["num_phones"])), dtype=np.float16)
                        batch_plabels_i = np.zeros((1,1), dtype=np.int32)
    
                        test_embeddings_1, test_embeddings_2 = sess.run([features_1, features_2], \
                                feed_dict={is_training: False, batch_speech: batch_speech_i, batch_labels: batch_labels_i, \
                                batch_pdata: batch_pdata_i, batch_plabels: batch_plabels_i})

                        feature_vector_1[batch_num, :] = test_embeddings_1[0]
                        feature_vector_2[batch_num, :] = test_embeddings_2[0]

                    np.save(os.path.join(savedata_dir)+"/data_embeddings_1_"+str(batch_file+1)+".npy", feature_vector_1.astype(np.float32))
                    np.save(os.path.join(savedata_dir)+"/data_embeddings_2_"+str(batch_file+1)+".npy", feature_vector_2.astype(np.float32))
                    shutil.copy(os.path.join(FLAGS.savedata_dir)+"/label_train_"+str(batch_file+1)+".npy", \
                            os.path.join(savedata_dir)+"/data_labels_"+str(batch_file+1)+".npy")
      
                    print("\n")

        shutil.copy(os.path.join(FLAGS.savedata_dir)+"/parameters.txt", os.path.join(savedata_dir)+"/parameters.txt")
        shutil.copy(os.path.join(FLAGS.savedata_dir)+"/data_size", os.path.join(savedata_dir)+"/data_size")
        shutil.copy(os.path.join(FLAGS.savedata_dir)+"/speaker_count", os.path.join(savedata_dir)+"/speaker_count")
        shutil.copy(os.path.join(FLAGS.savedata_dir)+"/speaker_list.npy", os.path.join(savedata_dir)+"/speaker_list.npy")
        shutil.copy(os.path.join(FLAGS.savedata_dir)+"/wavid_list.npy", os.path.join(savedata_dir)+"/wavid_list.npy")


def main(argv):

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_nums
    if len(FLAGS.device_nums) == 0:
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

    _test()

    print("### the DROP-TDNN AM-Softmax embedding was extracted.")


if __name__ == "__main__":
    app.run(main)
