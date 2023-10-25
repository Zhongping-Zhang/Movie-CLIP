import os,json
from os.path import join, basename, dirname
import pickle
import argparse
from tqdm import tqdm
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='val', choices=['train', 'test', 'val'])
parser.add_argument('--clip_version', default="clip",)
parser.add_argument('--model_version', default="ViT-L/14@336px")
parser.add_argument('--data_dir', default='data/trailer30K/MovieCLIP_features/Text_features')
parser.add_argument('--text_file', default="whisper_medium_asr_output.json")
parser.add_argument('--save_file',default="whisper_medium_asr_output_ViT-L-14@336px.pkl")
parser.add_argument('--cache_dir', type=str, default="/projectnb/ivc-ml/zpzhang/checkpoints/clip_cache")
args = parser.parse_args()

if args.clip_version=="eva_clip":
    from eva_clip import create_model_and_transforms, get_tokenizer
    model, _, preprocess = create_model_and_transforms(args.model_version, "eva_clip", force_custom_clip=True,
                                                   cache_dir=args.cache_dir)
    tokenizer = get_tokenizer(args.model_version)
    model = model.to(device)
elif args.clip_version=="clip":
    # ViT-B/32; ViT-B/16; # (1,512)
    # ViT-L/14; ViT-L/14@336px; # (1,768)
    import clip
    model,preprocess = clip.load(args.model_version, device=device, download_root=args.cache_dir)
    model = model.to(device)
model.eval()


# with open(join(args.data_root,"vocab/vocab_genres21.json"),"r") as f:
#     genres21 = json.load(f)
with open(join(args.data_dir,args.text_file),"r") as f:
    text_data = json.load(f)


def get_text_clip_features(text_list):
    text = clip.tokenize(text_list, context_length=77, truncate=True).to(device)
    text_features = model.encode_text(text)
    return text_features.cpu().detach().numpy()

text_feat_dict = {}

for movie_id, text_str in tqdm(text_data.items(),desc="extract text features"):
    text_feat = get_text_clip_features([text_str]) # (1,768)
    text_feat_dict[movie_id] = text_feat[0]

with open(join(args.data_dir,args.save_file),"wb") as handle:
    pickle.dump(text_feat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# text_list = list(genres21.keys())
# text_features = get_genre_clip_features(text_list)
# print('text_list',text_list)
# print("text_features",text_features.shape)
#
# args.model_version=args.model_version.replace("/","-")
# np.save(join(args.data_root,"vocab/{}_genre_features.npy".format(args.model_version)),text_features)
























