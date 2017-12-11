#####################################
#   import general use modeules     #
#####################################
import os 
import numpy as np

#####################################
#  import librosa related modeules  #
#####################################
import librosa

corpus_path = "vcc2016/vcc2016_training/"
speaker_list = ["SF1", "SF2", "SF3", "SM1", "SM2", "TF1", "TF2", "TM1", "TM2", "TM3"]
outdir = "data"

#####################################
#    iterate through speaker list   #
#####################################
data = []
for i, speaker in enumerate(speaker_list):
    speaker_path = corpus_path+speaker
    for j in range(1, 163):
        if j <= 9:
            wav = "10000{}.wav".format(j)
        elif j <= 99:
            wav = "1000{}.wav".format(j)
        else:
            wav = "100{}.wav".format(j)
        data.append((i, speaker_path+'/'+wav))

#####################################
#       check if outdir exists      #
#####################################
if not os.path.exists(outdir):
    os.makedirs(outdir)

#####################################
#    save labeled data to outdir    #
#####################################
with open(outdir+"/train_label.csv", "w", encoding="utf-8") as outfile:
    for i, y in data:
        outfile.write("{0},{1}\n".format(i, y))

#####################################
#    turn num to one-hot encoding   #
#####################################
def n2onehot(i, m):
    tmp = [0]*m
    tmp[i] = 1
    return np.array(tmp)

#####################################
#       load data from csv          #
#####################################
def load_data(filename):
    data = []
    total_label_num = 10
    with open(filename, "r") as infile:
        for line in infile:
            i, f = line.rstrip().split(',')
            y, _ = librosa.load(f)
            data.append((n2onehot(int(i), total_label_num), y))
    return np.array(data)

data = load_data("data/train_label.csv")
np.save("data/train_label.npy", data)
