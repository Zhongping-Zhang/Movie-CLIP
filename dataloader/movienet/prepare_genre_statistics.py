import argparse
import pandas as pd
from os.path import join, basename, dirname
import json
import os, glob
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_dir', type=str, default="data/MovieNet/meta", help='path to metadata files')
parser.add_argument('--trailer_info_dir', type=str, default="data/trailer30K/MovieCLIP_features/keyFrames_info", help="path to meta data")
parser.add_argument('--output_dir', type=str, default="data/trailer30K/MovieCLIP_features")
args = parser.parse_args()
print(args)



genres21 = ['Drama', 'Comedy', 'Thriller', 'Action', 'Romance',
            'Horror', 'Crime', 'Documentary', 'Adventure', 'Sci-Fi',
            'Family', 'Fantasy', 'Mystery', 'Biography', 'Animation',
            'History', 'Music', 'War', 'Sport', 'Musical', 'Western']

total_count = 0
short_trailer_count = 0
invalid_genre_count = 0

len_threshold = 12  # get rid of trailers which are less than 12 shots
trailer_shot_length_dict = {}
genre_stat = {}


file_list = glob.glob(args.trailer_info_dir + "/*.json")
for file_path in tqdm(file_list, desc='get valid trailers'):
    imdb_id = file_path.split("/")[-1].split(".")[0]
    with open(os.path.join(args.metadata_dir, imdb_id + ".json"), "r") as f:
        meta = json.load(f)
    genres = meta['genres']  # 1. genres

    if not genres:
        continue
    total_count += 1

    with open(file_path, 'r') as f:
        json_info = json.load(f)

    trailer_clips = json_info['clip']

    if len(trailer_clips) < len_threshold:
        short_trailer_count += 1
        continue

    flag = 0
    for genre in genres:
        if genre not in genres21: continue

        flag = 1
        if genre not in genre_stat:
            genre_stat[genre] = 1
        else:
            genre_stat[genre] += 1

    if flag == 0:
        invalid_genre_count += 1
        continue

    trailer_shot_length_dict[imdb_id] = len(trailer_clips)

print("total files number: %d" % len(file_list))
print("# trailers of invalid genres: %d" % invalid_genre_count)
print("# trailers of less than 12 shots: %d" % short_trailer_count)

os.makedirs(join(args.output_dir,"vocab"),exist_ok=True)
with open(os.path.join(args.output_dir, "vocab/trailer_shot_length>%d_info.json" % len_threshold), 'w') as f:
    json.dump(trailer_shot_length_dict, f, indent=2)


"""
total_number: 28717
# of valid genres: 28621
# of short trailer:
    less 12: 124  x
    less 13: 210
"""

sorted_genre_stat = {k: v for k, v in sorted(genre_stat.items(), key=lambda item: item[1])[::-1]}
with open(os.path.join(args.output_dir, "vocab/trailer_genre_statistics.json"), 'w') as fv:
    json.dump(sorted_genre_stat, fv, indent=2)

plt.figure()
plt.title("genre statistics")
plt.bar(sorted_genre_stat.keys(), sorted_genre_stat.values())
plt.xticks(rotation=70)
plt.savefig(os.path.join(args.output_dir, "vocab/trailer_genre_statistics.png"), bbox_inches='tight')

















