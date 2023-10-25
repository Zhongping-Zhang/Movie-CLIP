import json
import random
import os
from tqdm import tqdm
import argparse

def load_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="data/CondensedMovies/MovieCLIP_features")
parser.add_argument('--keyframe_file',type=str,default='imdbid2genre_frame.txt')
parser.add_argument('--save_genre_info',type=bool,default=False)
parser.add_argument('--convert2dict',type=bool,default=False)
args = parser.parse_args()
print(args)

genre_stat = load_json(os.path.join(args.data_dir,"vocab/genre_stat.json"))
genres21 = list(genre_stat.keys())

vocab_genres21 = {}
idx = 0
for genre in genres21:
    vocab_genres21[genre] = idx
    idx+=1
with open(os.path.join(args.data_dir, "vocab/vocab_genres21.json"),"w") as fv:
    json.dump(vocab_genres21,fv) # save vocab (21 genres)



with open(os.path.join(args.data_dir,"imdbid2genre_frame.txt"),"r") as f:
    samples = f.readlines()

imdb2clip = load_json(os.path.join(args.data_dir, "vocab/imdb2clip.json"))
clip_list = []
for clip in imdb2clip.values():
    clip_list+=clip
print("imdb list length: ",len(imdb2clip)) # 2803 imdb_id
print("clip list length: ", len(clip_list)) # 532176 frame samples => 22174 clips

random.Random(123).shuffle(clip_list)


train_len = int(len(clip_list)/10*7)
val_len = int(len(clip_list)/10)
test_len = len(clip_list)-train_len-val_len

clip_train_list = clip_list[:train_len]
clip_val_list = clip_list[train_len:(train_len+val_len)]
clip_test_list = clip_list[(train_len+val_len):]

genre_stat_train = {}
genre_stat_val = {}
genre_stat_test = {}
for genre in genres21:
    genre_stat_train[genre] = 0
    genre_stat_val[genre] = 0
    genre_stat_test[genre] = 0

f_train = open(os.path.join(args.data_dir,"key_frame_train.txt"),"w")
f_val = open(os.path.join(args.data_dir,"key_frame_val.txt"),"w")
f_test = open(os.path.join(args.data_dir,"key_frame_test.txt"),"w")

for sample in tqdm(samples, desc='split train/val/test set'):
    eles = sample.split("\t")
    img_subid = int(eles[1][-5])
    if img_subid==0 or img_subid==1: # 3 frames for each shot, skip the first two frames
        continue

    imdb_id = eles[0]
    clip_name = eles[1].split("/")[-1].split("shot")[0]
    genres = eles[2][2:-3].split("', '")


    genres_id = []
    flag = 0
    for genre in genres:
        if genre not in genres21: continue
        flag=1
        genres_id.append(str(vocab_genres21[genre]))

    assert flag==1
    genres_id_str = ' '.join(genres_id)


    if clip_name in clip_train_list:
        f_train.write(eles[0]+"\t"+eles[1]+"\t"+genres_id_str+"\n")
        for genre in genres:
            if genre not in genres21: continue
            genre_stat_train[genre] += 1/8 # scale by shot_number per trailer
    elif clip_name in clip_val_list:
        f_val.write(eles[0]+"\t"+eles[1]+"\t"+genres_id_str+"\n")
        for genre in genres:
            if genre not in genres21: continue
            genre_stat_val[genre] += 1/8
    elif clip_name in clip_test_list:
        f_test.write(eles[0]+"\t"+eles[1]+"\t"+genres_id_str+"\n")
        for genre in genres:
            if genre not in genres21: continue
            genre_stat_test[genre] += 1/8
    else:
        assert False

f_train.close()
f_val.close()
f_test.close()

if args.save_genre_info:

    os.makedirs(os.path.join(args.data_dir,"vocab/statistics"),exist_ok=True)
    sorted_genre_stat_train = {k: v for k, v in sorted(genre_stat_train.items(), key=lambda item: item[1])[::-1]}
    sorted_genre_stat_val = {k: v for k, v in sorted(genre_stat_val.items(), key=lambda item: item[1])[::-1]}
    sorted_genre_stat_test = {k: v for k, v in sorted(genre_stat_test.items(), key=lambda item: item[1])[::-1]}

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()
    plt.title("genre statistics train")
    plt.bar(sorted_genre_stat_train.keys(), sorted_genre_stat_train.values())
    plt.xticks(rotation=70)
    plt.savefig(os.path.join(args.data_dir, "vocab/statistics/trailer_genre_train.png"), bbox_inches='tight')

    plt.figure()
    plt.title("genre statistics val")
    plt.bar(sorted_genre_stat_val.keys(), sorted_genre_stat_val.values())
    plt.xticks(rotation=70)
    plt.savefig(os.path.join(args.data_dir, "vocab/statistics/trailer_genre_val.png"), bbox_inches='tight')

    plt.figure()
    plt.title("genre statistics test")
    plt.bar(sorted_genre_stat_test.keys(), sorted_genre_stat_test.values())
    plt.xticks(rotation=70)
    plt.savefig(os.path.join(args.data_dir, "vocab/statistics/trailer_genre_test.png"), bbox_inches='tight')

    with open(os.path.join(args.data_dir, "vocab/statistics/trailer_genre_train.json"), 'w') as fv:
        json.dump(sorted_genre_stat_train, fv)

    with open(os.path.join(args.data_dir, "vocab/statistics/trailer_genre_val.json"), 'w') as fv:
        json.dump(sorted_genre_stat_val, fv)

    with open(os.path.join(args.data_dir, "vocab/statistics/trailer_genre_test.json"), 'w') as fv:
        json.dump(sorted_genre_stat_test, fv)

if args.convert2dict:
    phase_list = ['train', 'val', 'test']

    for phase in phase_list:
        with open(os.path.join(args.data_dir, "key_frame_{}.txt".format(phase)), 'r') as f:
            shot_data = f.readlines()

        trailer_dict = {}

        for sample in tqdm(shot_data):
            eles = sample.split("\t")
            trailer_id = eles[0]
            movie_id = eles[1].split("/")[-1].split("shot")[0]
            shot_path = eles[1]
            shot_num = eles[1].split("shot")[-1].split("_")[0]

            key_frame0_path = shot_path[:-5] + "0.jpg"
            key_frame1_path = shot_path[:-5] + "1.jpg"
            key_frame2_path = shot_path[:-5] + "2.jpg"

            assert shot_path == key_frame2_path

            if movie_id not in trailer_dict:
                trailer_dict[movie_id] = {}

            trailer_dict[movie_id][shot_num] = [key_frame0_path, key_frame1_path, key_frame2_path]
            trailer_dict[movie_id]['label'] = eles[2]

        with open(os.path.join(args.data_dir, "data_{}.json".format(phase)), 'w') as f:
            json.dump(trailer_dict, f)
