import json
import random
import os
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="data/CondensedMovies/MovieCLIP_features", help='path to trailer30K files')
parser.add_argument('--keyframe_file',type=str,default='imdbid2genre_frame.txt')
parser.add_argument('--vocab_dir',type=str,default='data/CondensedMovies/MovieCLIP_features/vocab')
args = parser.parse_args()
print(args)

genres21 = ['Drama', 'Comedy', 'Thriller', 'Action', 'Romance',
            'Horror', 'Crime', 'Documentary', 'Adventure', 'Sci-Fi',
            'Family', 'Fantasy', 'Mystery', 'Biography', 'Animation',
            'History', 'Music', 'War', 'Sport', 'Musical', 'Western'] # selective genres, consistent with MovieNet

with open(os.path.join(args.data_dir,args.keyframe_file),"r") as f:
    samples = f.readlines()

genre_stat = {}
imdb2clip = {}

for sample in tqdm(samples):
    eles = sample.split("\t")
    imdb_id = eles[0]
    clip_name = eles[1].split("/")[-1].split("shot")[0]
    genres = eles[2][2:-3].split("', '")
    # print(genres)

    assert len(clip_name)==12

    if imdb_id not in imdb2clip:
        imdb2clip[imdb_id] = [clip_name]
    if clip_name not in imdb2clip[imdb_id]:
        imdb2clip[imdb_id].append(clip_name)

    flag = 0
    for genre in genres:
        if genre not in genres21: continue
        flag =1
        if genre not in genre_stat:
            genre_stat[genre] = 1/24
        else:
            genre_stat[genre] += 1/24

sorted_genre_stat = {k: v for k, v in sorted(genre_stat.items(), key=lambda item: item[1])[::-1]}
imdb2clip = {k: v for k, v in sorted(imdb2clip.items(), key=lambda item: len(item[1]))[::-1]}


os.makedirs(args.vocab_dir, exist_ok=True)
with open(os.path.join(args.vocab_dir, "imdb2clip.json"),"w") as f:
    json.dump(imdb2clip,f)
with open(os.path.join(args.vocab_dir,"genre_stat.json"), "w") as fs:
    json.dump(sorted_genre_stat, fs)

plt.figure()
plt.title("genre statistics")
plt.bar(sorted_genre_stat.keys(),sorted_genre_stat.values())
plt.xticks(rotation=70)
plt.savefig(os.path.join(args.vocab_dir,"genre_stat_plot.png"),bbox_inches='tight')




