import spacy
from os.path import join, basename, dirname
import os
import json
from string import punctuation
from collections import Counter
from tqdm import tqdm
import argparse


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_entities", after="ner") # merge neighbor word


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="data/trailer30K/MovieCLIP_features/Text_features", help='path to trailer30K files')
parser.add_argument('--caption_file', type=str, default="whisper_medium_asr_output.json", help="path to caption file")
parser.add_argument('--save_name', type=str, default="whisper_medium_keywords.json", help="save name")
parser.add_argument('--top_k', type=int, default=10, help="top k hotwords")
args = parser.parse_args()
print(args)


def get_keywords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    # A proper noun is a noun (or nominal content word) that is the name (or part of the name) of a specific individual, place, or object.
    doc = nlp(text.lower())
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result

def textlist_lower(text_list):
    for i in range(len(text_list)):
        text_list[i] = text_list[i].lower()
    return text_list

with open(join(args.data_dir,args.caption_file),"r") as f:
    caption_data = json.load(f)


genre_list = ["Drama", "Comedy", "Thriller", "Action", "Romance", "Horror", "Crime", "Documentary",
              "Adventure", "Sci-Fi", "Family", "Fantasy", "Mystery", "Biography", "Animation", "History",
              "Music", "War", "Sport", "Musical", "Western"]
genre_list = textlist_lower(genre_list)
print(genre_list)

keyword_dict = {}

whisper_num = 0
consistent_num = 0
inconsistent_num = 0

whisper_consistent_num = 0
whisper_inconsistent_num = 0


for imdb_id, text_str in tqdm(caption_data.items()):
    if "MovieNet" in args.meta_ROOT:
        with open(join(args.meta_ROOT,imdb_id+".json"),"r") as f:
            meta = json.load(f)
        gt_genres = meta['genres']
        if not gt_genres: continue
        textlist_lower(gt_genres)
    # print(gt_genres)


    keywords = get_keywords(text_str)
    hashtags = [x[0] for x in Counter(keywords).most_common(args.top_k)]



    if "MovieNet" in args.meta_ROOT:
        for genre in genre_list:
            if genre in hashtags:
                # print(genre)
                # print(imdb_id)
                if genre in gt_genres:
                    consistent_num+=1
                    if "youtube" in text_str[:10]:
                        youtube_consistent_num+=1
                    elif "silero" in text_str[:10]:
                        silero_consistent_num+=1
                    else:
                        pass
                        #assert False
                else:
                    inconsistent_num+=1
                    if "youtube" in text_str[:10]:
                        youtube_inconsistent_num+=1
                    elif "silero" in text_str[:10]:

                        keyword_dict[imdb_id] = hashtags

with open(join(args.data_dir,args.save_name),"w") as fk:
    json.dump(keyword_dict, fk)

print("youtube number: ", youtube_num)
print("silero number: ", silero_num)
print("consistent number: ", consistent_num)
print("inconsistent number", inconsistent_num)
print("youtube consistent:", youtube_consistent_num)
print("silero consistent:", silero_consistent_num)
print("youtube_inconsistent:", youtube_inconsistent_num)
print("silero inconsistent:", silero_inconsistent_num)