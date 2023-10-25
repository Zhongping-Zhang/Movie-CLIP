import os,glob
import json
import random
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="data/trailer30K/MovieCLIP_features", help='path to trailer30K files')
parser.add_argument('--metadata_dir', type=str, default="data/MovieNet/meta", help="path to meta data")
parser.add_argument('--keyframe_dir',type=str,default="data/trailer30K/MovieCLIP_features/keyFrames")
parser.add_argument('--trailer_info_file',type=str,default="trailer_shot_length>12_info.json")
parser.add_argument('--save_genre_info',type=bool,default=False)
parser.add_argument('--convert2dict',type=bool,default=False)
args = parser.parse_args()
print(args)

with open(os.path.join(args.data_dir,"key_frame_info.txt"),"r") as f:
    samples = f.readlines()

genres21 = ['Drama', 'Comedy', 'Thriller', 'Action', 'Romance',
            'Horror', 'Crime', 'Documentary', 'Adventure', 'Sci-Fi',
            'Family', 'Fantasy', 'Mystery', 'Biography', 'Animation',
            'History', 'Music', 'War', 'Sport', 'Musical', 'Western']

vocab_genres21 = {}
idx = 0
for genre in genres21:
    vocab_genres21[genre] = idx
    idx+=1
with open(os.path.join(args.data_dir, "vocab/vocab_genres21.json"),"w") as fv:
    json.dump(vocab_genres21,fv,indent=2)

with open(os.path.join(args.data_dir,"vocab",args.trailer_info_file),'r') as ft:
    trailer_dict = json.load(ft)
    trailer_list = list(trailer_dict.keys())


random.Random(123).shuffle(trailer_list)

train_len = int(len(trailer_list)/10*7) # 227752 shots, 28469 trailers
val_len = int(len(trailer_list)/10)
test_len = len(trailer_list)-train_len-val_len
trailer_train_list = trailer_list[:train_len]
trailer_val_list = trailer_list[train_len:(train_len+val_len)]
trailer_test_list = trailer_list[(train_len+val_len):]


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
    sample = sample.split("\n")[0]
    eles = sample.split("\t")
    trailer_id = eles[0]
    genres = eles[2].split(" ")

    genres_id = []
    flag = 0
    for genre in genres:
        if genre not in genres21: continue
        flag=1
        genres_id.append(str(vocab_genres21[genre]))

    assert flag==1
    genres_id_str = ' '.join(genres_id)

    eles[1] = eles[1].replace("/research/zpzhang/DATA/trailer30K/trailer_key_frames",args.keyframe_dir)

    if trailer_id in trailer_train_list:
        f_train.write(eles[0]+"\t"+eles[1]+"\t"+genres_id_str+"\n")
        for genre in genres:
            if genre not in genres21: continue
            genre_stat_train[genre] += 1/8 # scale by shot_number per trailer
    elif trailer_id in trailer_val_list:
        f_val.write(eles[0]+"\t"+eles[1]+"\t"+genres_id_str+"\n")
        for genre in genres:
            if genre not in genres21: continue
            genre_stat_val[genre] += 1/8
    elif trailer_id in trailer_test_list:
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

    plt.figure()
    plt.title("genre statistics train")
    plt.bar(sorted_genre_stat_train.keys(),sorted_genre_stat_train.values())
    plt.xticks(rotation=70)
    plt.savefig(os.path.join(args.data_dir,"vocab/statistics/trailer_genre_train.png"),bbox_inches='tight')

    plt.figure()
    plt.title("genre statistics val")
    plt.bar(sorted_genre_stat_val.keys(),sorted_genre_stat_val.values())
    plt.xticks(rotation=70)
    plt.savefig(os.path.join(args.data_dir,"vocab/statistics/trailer_genre_val.png"),bbox_inches='tight')

    plt.figure()
    plt.title("genre statistics test")
    plt.bar(sorted_genre_stat_test.keys(),sorted_genre_stat_test.values())
    plt.xticks(rotation=70)
    plt.savefig(os.path.join(args.data_dir,"vocab/statistics/trailer_genre_test.png"),bbox_inches='tight')

    with open(os.path.join(args.data_dir,"vocab/statistics/trailer_genre_train.json"),'w') as fv:
        json.dump(sorted_genre_stat_train,fv)

    with open(os.path.join(args.data_dir,"vocab/statistics/trailer_genre_val.json"),'w') as fv:
        json.dump(sorted_genre_stat_val,fv)

    with open(os.path.join(args.data_dir,"vocab/statistics/trailer_genre_test.json"),'w') as fv:
        json.dump(sorted_genre_stat_test,fv)

if args.convert2dict:
    phase_list = ['train', 'val', 'test']

    for phase in phase_list:
        with open(os.path.join(args.data_dir, "key_frame_{}.txt".format(phase)), 'r') as f:
            shot_data = f.readlines()

        trailer_dict = {}

        for sample in tqdm(shot_data):
            eles = sample.split("\t")
            movie_id = eles[0]
            shot_path = eles[1]
            shot_num = eles[1].split("shot")[-1].split("_")[1]

            key_frame0_path = shot_path[:-5] + "0.jpg"
            key_frame1_path = shot_path[:-5] + "1.jpg"
            key_frame2_path = shot_path[:-5] + "2.jpg"

            assert shot_path == key_frame2_path

            if movie_id not in trailer_dict:
                trailer_dict[movie_id] = {}

            trailer_dict[movie_id][shot_num] = [key_frame0_path, key_frame1_path, key_frame2_path]
            trailer_dict[movie_id]['label'] = eles[2]

        with open(os.path.join(args.data_dir, "data_{}.json".format(phase)), 'w') as f:
            json.dump(trailer_dict, f, indent=2)





