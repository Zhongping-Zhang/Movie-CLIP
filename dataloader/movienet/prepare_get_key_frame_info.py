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

args = parser.parse_args()
print(args)



with open(os.path.join(args.data_dir, "vocab", args.trailer_info_file),'r') as f:
    trailer_dict = json.load(f)


sub_list = ["sub%d"%i for i in range(3)]
keyf_list = []
for subfolder in sub_list:
    keyf_list += glob.glob(os.path.join(args.keyframe_dir,subfolder,'shot_keyf')+"/*.json") # we separate the key frames into 3 folders (sub0, sub1, sub2)

path_dict = {}
for keyf in keyf_list:
    imdb_id = keyf.split("/")[-1].split("_")[0]
    if imdb_id in trailer_dict:
        path_dict[imdb_id] = keyf.split(imdb_id)[0]+imdb_id

with open(os.path.join(args.data_dir,"vocab/imdb2keyframefolder.json"),'w') as fp:
    json.dump(path_dict,fp,indent=2)

f_save_txt = open(os.path.join(args.data_dir,"key_frame_info_all.txt"),"w")
for imdb_id,value in tqdm(trailer_dict.items()):
    shot_list = random.sample([i for i in range(value)][2:-2], 8)
    shot_list_sorted = sorted(shot_list)

    trailer_path = path_dict[imdb_id]
    with open(os.path.join(args.metadata_dir, imdb_id+".json"),"r") as f:
        meta = json.load(f)
        genres = meta['genres'] # 1. genres
        genres_str = ' '.join(genres)

    for shot_num in shot_list_sorted:
        img_save_name = os.path.join(trailer_path,"shot_%.4d_img_2.jpg"%shot_num)
        f_save_txt.write(imdb_id+"\t"+img_save_name+"\t"+genres_str+"\n")

f_save_txt.close()



