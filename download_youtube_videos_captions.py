import json
from pytube import YouTube
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video_ROOT', type=str, default=None, help='path to save source videos')
parser.add_argument('--caption_ROOT', type=str, default=None, help='path to save captions')
args = parser.parse_args()
print(args)


os.makedirs(args.video_ROOT,exist_ok=True)
os.makedirs(args.caption_ROOT,exist_ok=True)

with open("data/trailer30K/trailer.urls.unique30K.v1.json",'r') as f:
    trailer30k = json.load(f) # 32753 trailers
    
    
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


trailer30k_dict = {}
for i in range(len(trailer30k)):
    sample = trailer30k[i]
    trailer30k_dict[sample['imdb_id']] = sample['youtube_id']



# implement the following code if you would like to collect captions from YouTube
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

with open(os.path.join(args.caption_ROOT,"cap_youtube.json"),'w') as f:
    json.dump(trailer_caps,f)
    
    



