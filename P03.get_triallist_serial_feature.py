import sys
import os
import json
import math
import numpy as np
import random
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io import savemat, loadmat

from absl import app
from absl import flags

from datetime import datetime

FLAGS = flags.FLAGS

flags.DEFINE_string('savemodel_dir', './exp/model', 'savemodel_dir.')
flags.DEFINE_string('savemodel_name', 'kaldi_dc_ecapa_tdnn_wo_PEL_amsoftmax_model_voxceleb2_train_mfcc80_2M_4gpus_batch128_clr', 'savemodel_name.')

flags.DEFINE_string('trainembedding_name', 'voxceleb2_train_combined_no_sil', 'trainembedding_name.')
flags.DEFINE_string('testembedding_name', 'voxceleb1_test_combined_no_sil', 'testembedding_name.')

## trial list = {trials_test_VoxCeleb1_cleaned, trials_all_VoxCeleb1-E_cleaned, trials_hard_VoxCeleb1-H_cleaned}
flags.DEFINE_string('triallist_dir', '/datatank/public_data/VoxCeleb_Dataset/vox2/trials_test_VoxCeleb1_cleaned', 'trial list file.')
## trial list = {trial-core-core_OK, trial-core-multi_OK, trial-assist-core_OK, trial-assist-multi_OK}
#flags.DEFINE_string('triallist_dir', '/datatank/public_data/SITW_Dataset/sitw_database.v4/eval/trials/trial-core-core_OK', 'trial list file.')
## trial list = {trials_cn-celeb}
#flags.DEFINE_string('triallist_dir', '/datatank/public_data/CN-Celeb/trials_cn-celeb', 'trial list file.')
## trial list (OTHERS)
#flags.DEFINE_string('triallist_dir', '/datatank/public_data/VoxCeleb_Dataset/voxsrc_2021/VoxSRC2021/data/verif/trials.txt', 'trial list file.')

flags.DEFINE_integer('load_steps', 0, 'load_steps.')
flags.DEFINE_boolean('prepare_data_yn', False, 'whether prepare training data.')
flags.DEFINE_boolean('use_augmented_data_yn', False, 'whether use augmented data.')

flags.DEFINE_string('embedding_type', '1', '"1","2"')

#-----------------------------------------
# triallist_type
# 1: id10270-x6uYqmx31kE-00001 id10270-8jEAjG6SegY-00008 target
# 2: 1 id10912/H5qe-mHhOyQ/00007.wav id10912/nDpaVYtKQUo/00004.wav
flags.DEFINE_string('triallist_type', '1', '"1","2"')
#-----------------------------------------

#data_num_perfile = 100000 # The amount of data saved per file.
flags.DEFINE_integer('data_num_perfile', 100000, 'data_num_perfile.')

data_augs_all = [ "babble", "music", "noise", "reverb" ]


def load_data(data_dir):

    bgmodel_dir = os.path.join(FLAGS.savemodel_dir, FLAGS.savemodel_name)

    print("### data_dir =", data_dir)
  
    file_list_all = np.array([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
    file_list = np.array([file_list_all[idx] for idx in range(file_list_all.shape[0]) \
            if file_list_all[idx].find("data_embeddings{}_".format(FLAGS.embedding_type)) >= 0])
  
    save_count = file_list.shape[0]
  
    print("### save_count =", save_count)

    fp = open(os.path.join(data_dir)+"/data_size", "r")
    line = fp.readline()
    fp.close()

    data_size = int(line)

    print("### data_size =", data_size)

    para_dict = {}

    fp = open(os.path.join(bgmodel_dir)+"/parameters.txt", "r")
    for line in iter(fp):
        temp_list = line.split("=")
        if len(temp_list) == 2:
            para_dict[temp_list[0]] = temp_list[1].replace("\n","")
    fp.close()
  
    feature_vector = np.zeros((data_size, int(para_dict["embed_dim"])), dtype=np.float32)
    label_vector = np.zeros((data_size,), dtype=np.int32)
  
    load_count = 0
  
    for save_index in range(save_count):
    
        data_embed_dir = os.path.join(data_dir)+"/data_embeddings{}_".format(FLAGS.embedding_type)+str(save_index+1)+".npy"
        label_embed_dir = os.path.join(data_dir)+"/data_labels_"+str(save_index+1)+".npy"
    
        if os.path.isfile(data_embed_dir) and os.path.isfile(label_embed_dir):
    
            data_embed = np.load(data_embed_dir)
            label_embed = np.load(label_embed_dir)
    
            print("### (", str(save_index+1), "/", str(save_count), ") data loading...", end="\t\r")
    
            index_s = load_count
            index_e = load_count + label_embed.shape[0]
    
            feature_vector[index_s:index_e,:] = data_embed
            label_vector[index_s:index_e] = label_embed
    
            load_count = load_count + label_embed.shape[0]

            os.remove(data_embed_dir)
            #os.remove(label_embed_dir)

        else:
            print("### ERROR: data (", "data_embeddings{}_".format(FLAGS.embedding_type)+str(save_index+1)+".npy", ") not found.")
            exit()
    
    feature_vector = feature_vector[:load_count]
    label_vector = label_vector[:load_count]
  
    print("\n")
    print(">>> feature_vector.shape =", feature_vector.shape)
    print(">>> label_vector.shape =", label_vector.shape)
    print("\n")
  
    return feature_vector, label_vector


def save_trial_embeddings():

    bgmodel_dir = os.path.join(FLAGS.savemodel_dir, FLAGS.savemodel_name)
    traindata_dir = os.path.join(bgmodel_dir, "embeddings", FLAGS.trainembedding_name + "-" + str(FLAGS.load_steps))
    testdata_dir = os.path.join(bgmodel_dir, "embeddings", FLAGS.testembedding_name + "-" + str(FLAGS.load_steps))

    if not os.path.isfile(FLAGS.triallist_dir):
        print("### error: trial list is not found.")
        print(">>>", FLAGS.triallist_dir)
        exit()
  
    if not os.path.exists(traindata_dir):
        print("### error: train data file is not found.")
        print(">>>", traindata_dir)
        exit()
  
    if not os.path.exists(testdata_dir):
        print("### error: save data file is not found.")
        print(">>>", testdata_dir)
        exit()
  
    para_dict = {}

    fp = open(os.path.join(bgmodel_dir)+"/parameters.txt", "r")
    for line in iter(fp):
        temp_list = line.split("=")
        if len(temp_list) == 2:
            para_dict[temp_list[0]] = temp_list[1].replace("\n","")
    fp.close()

    triallist_name = FLAGS.triallist_dir.split("/")[-1]
    print("### read trial list:", triallist_name)
  
    lines_num = sum(1 for line in open(FLAGS.triallist_dir))
   
    if FLAGS.prepare_data_yn:
  
        ### load training embeddings
        feature_tr_vector, label_tr_vector = load_data(traindata_dir)
    
        #np.save(os.path.join(traindata_dir)+"/data_embeddings{}.npy".format(FLAGS.embedding_type), feature_tr_vector)
        #np.save(os.path.join(traindata_dir)+"/data_labels.npy", label_tr_vector)

        if label_tr_vector.shape[0] > 0:
            mdic = {"data": feature_tr_vector}
            savemat(os.path.join(traindata_dir)+"/data_embeddings{}.mat".format(FLAGS.embedding_type), mdic)
            mdic = {"data": label_tr_vector}
            savemat(os.path.join(traindata_dir)+"/data_labels.mat", mdic)
      
        ### load testing embeddings
        if traindata_dir != testdata_dir:
            feature_te_vector, label_te_vector = load_data(testdata_dir)
        else:
            feature_te_vector = feature_tr_vector
            label_te_vector = label_tr_vector
    
        #np.save(os.path.join(testdata_dir)+"/data_embeddings{}.npy".format(FLAGS.embedding_type), feature_te_vector)
        #np.save(os.path.join(testdata_dir)+"/data_labels.npy", label_te_vector)

        if label_te_vector.shape[0] > 0:
            mdic = {"data": feature_te_vector}
            savemat(os.path.join(testdata_dir)+"/data_embeddings{}.mat".format(FLAGS.embedding_type), mdic)
            mdic = {"data": label_te_vector}
            savemat(os.path.join(testdata_dir)+"/data_labels.mat", mdic)
  
    #else:
    if 1 == 1:
      
        #feature_tr_vector = np.load(os.path.join(traindata_dir)+"/data_embeddings{}.npy".format(FLAGS.embedding_type))
        #label_tr_vector = np.load(os.path.join(traindata_dir)+"/data_labels.npy")
        #feature_te_vector = np.load(os.path.join(testdata_dir)+"/data_embeddings{}.npy".format(FLAGS.embedding_type))
        #label_te_vector = np.load(os.path.join(testdata_dir)+"/data_labels.npy")

        feature_tr_vector = loadmat(os.path.join(traindata_dir)+"/data_embeddings{}.mat".format(FLAGS.embedding_type))["data"]
        label_tr_vector = loadmat(os.path.join(traindata_dir)+"/data_labels.mat")["data"]
        feature_te_vector = loadmat(os.path.join(testdata_dir)+"/data_embeddings{}.mat".format(FLAGS.embedding_type))["data"]
        label_te_vector = loadmat(os.path.join(testdata_dir)+"/data_labels.mat")["data"]

    if not FLAGS.use_augmented_data_yn:

        print("\n")
        now = datetime.now()
        print(">>> date and time =", now.strftime("%d/%m/%Y %H:%M:%S"))

        sel_index_list = []

        wavid_list = np.load(os.path.join(traindata_dir)+"/wavid_list.npy")
        for proc_i in range(wavid_list.shape[0]):
            if proc_i % 10000 == 0 or proc_i+1 == wavid_list.shape[0]:
                print(">>> proc_i =", proc_i, ", wavid =", wavid_list[proc_i], "\t", end="\r")

            subfields = wavid_list[proc_i].split("-")

            if not subfields[len(subfields)-1] in data_augs_all:
                sel_index_list.append(proc_i)

        print("\n")
        now = datetime.now()
        print(">>> date and time =", now.strftime("%d/%m/%Y %H:%M:%S"))

        sel_index_list = np.asarray(sel_index_list)
        feature_tr_vector = feature_tr_vector[sel_index_list]
        label_tr_vector = label_tr_vector[:, sel_index_list]

        print(">>> sel_index_list.shape =", sel_index_list.shape)
        print(">>> feature_tr_vector.shape =", feature_tr_vector.shape)
        print(">>> label_tr_vector.shape =", label_tr_vector.shape)

        if label_tr_vector.shape[0] > 0 and FLAGS.prepare_data_yn:
            mdic = {"data": feature_tr_vector}
            savemat(os.path.join(traindata_dir)+"/data_embeddings_origin{}.mat".format(FLAGS.embedding_type), mdic)
            mdic = {"data": label_tr_vector}
            savemat(os.path.join(traindata_dir)+"/data_labels_origin.mat", mdic)
  
  
    check_count = 0
    save_count = 0
  
    print("\n")
  
#    pair_embedding_1 = np.zeros((lines_num, int(para_dict["embed_dim"])), dtype=np.float32)
#    pair_embedding_2 = np.zeros((lines_num, int(para_dict["embed_dim"])), dtype=np.float32)
#    pair_label = np.zeros((lines_num,), dtype=np.int32)
    pair_ind = np.zeros((lines_num, 3), dtype=np.int32)
  
    wavid_list = np.load(os.path.join(testdata_dir)+"/wavid_list.npy")
  
    if feature_te_vector.shape[0] != wavid_list.shape[0]:

        print(">>> feature_te_vector.shape =", feature_te_vector.shape)
        print(">>> wavid_list.shape =", wavid_list.shape)
        print("### ERROR: The number of features is not equal to the number of wavid list.")
        exit()
  
    else:
  
        with open(FLAGS.triallist_dir) as fp:
            for line in fp:
                line = (line.rstrip().replace("\r","")).split(" ")
                if check_count % 1000 == 0 or check_count+1 == lines_num:
                    print("(", check_count+1,"/", lines_num,") loading:", line, end="\t\r")
  
                if FLAGS.triallist_type == "1":
                    #id10270-x6uYqmx31kE-00001 id10270-8jEAjG6SegY-00008 target
                    #id10270-x6uYqmx31kE-00001 id10300-ize_eiCFEg0-00003 nontarget
                    enroll_name = line[0]
                    test_name = line[1]
                    eval_target = line[2] if len(line) > 2 else "none"
                else:
                    #1 id10560/p_V0oeCcc0w/00011.wav id10560/_SIZKabFLAM/00001.wav
                    #0 id10792/La6IDPsWJHE/00007.wav id11044/BVzEyYdVLXg/00001.wav
                    enroll_name = line[1].replace("/","-").replace(".wav","")
                    test_name = line[2].replace("/","-").replace(".wav","")
                    eval_target = "target" if int(line[0]) == 1 else "nontarget"
  
                sel_enroll_index = np.where(wavid_list == enroll_name)[0]
                pair_ind[save_count, 0] = sel_enroll_index
                sel_test_index = np.where(wavid_list == test_name)[0]
                pair_ind[save_count, 1] = sel_test_index
  
                #print(">>> sel_enroll_index =", sel_enroll_index)
                #print(">>> sel_test_index =", sel_test_index)
  
                if sel_enroll_index.shape[0] > 0 and sel_test_index.shape[0] > 0:
  
#                    sel_enroll_embedding = feature_te_vector[sel_enroll_index][0]
#                    sel_test_embedding = feature_te_vector[sel_test_index][0]
#  
#                    pair_embedding_1[save_count] = sel_enroll_embedding
#                    pair_embedding_2[save_count] = sel_test_embedding
                    if eval_target == "target":
#                        pair_label[save_count] = 1
                        pair_ind[save_count, 2] = 1

                    if eval_target == "none":
                        pair_ind[save_count, 2] = -1
  
                    save_count = save_count + 1

                else:
                    print("### ERROR: wavid mismatch error.")
                    exit()
  
                check_count = check_count + 1
  
#        pair_embedding_1 = pair_embedding_1[:save_count]
#        pair_embedding_2 = pair_embedding_2[:save_count]
#        pair_label = pair_label[:save_count]
        pair_ind = pair_ind[:save_count]
 
        #np.save(os.path.join(testdata_dir)+"/pair_embedding{}_".format(FLAGS.embedding_type)+triallist_name+"_1.npy", pair_embedding_1)
        #np.save(os.path.join(testdata_dir)+"/pair_embedding{}_".format(FLAGS.embedding_type)+triallist_name+"_2.npy", pair_embedding_2)
        #np.save(os.path.join(testdata_dir)+"/pair_label{}_".format(FLAGS.embedding_type)+triallist_name+".npy", pair_label)

#        num_perfiles = math.ceil(save_count / data_num_perfile)
#
#        if num_perfiles > 1:
#            for proc_i in range(num_perfiles):
#                data_s = proc_i * data_num_perfile
#                data_e = data_s + data_num_perfile
#                if data_e > save_count:
#                    data_e = save_count
#                mdic = {"data": pair_embedding_1[data_s:data_e]}
#                savemat(os.path.join(testdata_dir)+"/pair_embedding{}_".format(FLAGS.embedding_type)+triallist_name+\
#                        "_1_part{}.mat".format(str(proc_i+1)), mdic)
#                mdic = {"data": pair_embedding_2[data_s:data_e]}
#                savemat(os.path.join(testdata_dir)+"/pair_embedding{}_".format(FLAGS.embedding_type)+triallist_name+\
#                        "_2_part{}.mat".format(str(proc_i+1)), mdic)
#                mdic = {"data": pair_label[data_s:data_e]}
#                savemat(os.path.join(testdata_dir)+"/pair_label{}_".format(FLAGS.embedding_type)+triallist_name+\
#                        "_part{}.mat".format(str(proc_i+1)), mdic)
#        else:
#            mdic = {"data": pair_embedding_1}
#            savemat(os.path.join(testdata_dir)+"/pair_embedding{}_".format(FLAGS.embedding_type)+triallist_name+"_1.mat", mdic)
#            mdic = {"data": pair_embedding_2}
#            savemat(os.path.join(testdata_dir)+"/pair_embedding{}_".format(FLAGS.embedding_type)+triallist_name+"_2.mat", mdic)
#            mdic = {"data": pair_label}
#            savemat(os.path.join(testdata_dir)+"/pair_label{}_".format(FLAGS.embedding_type)+triallist_name+".mat", mdic)
        mdic = {"data": pair_ind}
        savemat(os.path.join(testdata_dir)+"/pair_ind_".format(FLAGS.embedding_type)+triallist_name+".mat", mdic)

#        pair_score = np.zeros((pair_label.shape[0],))
        pair_score = np.zeros((pair_ind.shape[0],))
  
#        for pair_index in range(pair_label.shape[0]):
        for pair_index in range(pair_ind.shape[0]):
#            score = cosine_similarity(np.expand_dims(pair_embedding_1[pair_index,:], axis=0), np.expand_dims(pair_embedding_2[pair_index,:], axis=0))
            score = cosine_similarity( \
                    np.expand_dims(feature_te_vector[pair_ind[pair_index, 0]], axis=0), \
                    np.expand_dims(feature_te_vector[pair_ind[pair_index, 1]], axis=0))
            pair_score[pair_index] = score
  
        #np.save(os.path.join(testdata_dir)+"/pair_score{}_".format(FLAGS.embedding_type)+triallist_name+".npy", pair_score)

        mdic = {"data": pair_score}
        savemat(os.path.join(testdata_dir)+"/pair_score{}_".format(FLAGS.embedding_type)+triallist_name+".mat", mdic)
  
        print("\n")
  
        print("### saved ###")
  
        print("\n")

def main(argv):

    if len(FLAGS.embedding_type) > 0:
        FLAGS.embedding_type = "_" + FLAGS.embedding_type

    save_trial_embeddings()
  
    print("### finish. ###")

if __name__ == "__main__":
    app.run(main)

