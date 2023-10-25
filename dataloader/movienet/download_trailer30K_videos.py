import json
from pytube import YouTube
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str,default="data/trailer30K")
parser.add_argument('--trailer_url_file',type=str,default="trailer.urls.unique30K.v1.json")
parser.add_argument('--video_ROOT', type=str, default="data/trailer30K/trailer30K_origin_videos", help='path to raw videos')
parser.add_argument('--caption_ROOT', type=str, default="data/trailer30K/trailer30K_caption", help='path to raw captions')
args = parser.parse_args()
print(args)


os.makedirs(args.video_ROOT,exist_ok=True)
os.makedirs(args.caption_ROOT,exist_ok=True)

with open(os.path.join(args.data_dir,args.trailer_url_file),'r') as f:
    trailer30k = json.load(f) # 32753 trailers


# Download videos
invalid_links = 0
for idx in tqdm(range(len(trailer30k))):
    sample = trailer30k[idx]
    imdb_id = sample['imdb_id']
    youtube_id = sample['youtube_id']

    os.makedirs(os.path.join(args.video_ROOT,imdb_id),exist_ok=True)
    video = YouTube('https://www.youtube.com/watch?v='+youtube_id)
    try:
        video.streams.filter(res="360p")[0].download(os.path.join(args.video_ROOT,imdb_id),filename=imdb_id)
    except:
        invalid_links+=1
        # print("skip "+imdb_id)
        os.removedirs(os.path.join(args.video_ROOT,imdb_id))
print("%d links unavailable"%invalid_links)


# Download captions
trailer30k_dict = {}
for i in range(len(trailer30k)):
    sample = trailer30k[i]
    trailer30k_dict[sample['imdb_id']] = sample['youtube_id']
    
from youtube_transcript_api import YouTubeTranscriptApi
num_caption = 0
valid_trailer_list = os.listdir(args.video_ROOT) # 28720 trailers
trailer_caps = {}

for imdb_id in tqdm(valid_trailer_list, ascii=True, desc='download captions'):
    youtube_id = trailer30k_dict[imdb_id]

    video = YouTube('https://www.youtube.com/watch?v='+youtube_id)
    try:
        srt = YouTubeTranscriptApi.get_transcript(youtube_id)
        num_caption+=1
    except:
        continue
    trailer_caps[imdb_id] = srt
print("%d captions"%num_caption)


