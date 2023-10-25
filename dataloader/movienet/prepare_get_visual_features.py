import os
from os.path import join, basename, dirname
import pickle as pkl
import argparse
from tqdm import tqdm
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='val', choices=['train', 'test', 'val'])
parser.add_argument('--clip_version', default="clip",)
parser.add_argument('--model_version', default="ViT-L/14@336px")
parser.add_argument('--data_root', default='data/trailer30K/MovieCLIP_features')
parser.add_argument('--cache_dir', type=str, default="/projectnb/ivc-ml/zpzhang/checkpoints/clip_cache")
args = parser.parse_args()



if args.clip_version=="eva_clip":
    # best version: EVA02_CLIP_E_psz14_plus_s9B
    # All versions:
    # EVA02-CLIP-B-16.json ==> image embeddings:(1,512)
    # EVA02-CLIP-L-14-336.json  EVA02-CLIP-L-14.json
    # EVA02-CLIP-bigE-14-plus.json  EVA02-CLIP-bigE-14.json
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

def get_image_clip_features(image_pil, normalize=False):
    # image_pil = Image.open(image_path).convert("RGB")
    image = preprocess(image_pil).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    if normalize:
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().detach().numpy()



with open(join(args.data_root,"key_frame_{}.txt".format(args.phase)),"r") as f:
    data = f.readlines()

SAVE_FOLDER = join(args.data_root,"Visual_features")
os.makedirs(SAVE_FOLDER,exist_ok=True)

feat_dict = {}
special_shot = 0

for index in tqdm(range(len(data)), desc='extract clip features'):
    sample = data[index]
    eles = data[index].split("\t")

    imdb_id = eles[0]

    shot_path = eles[1]
    shot_frame0_path = shot_path[:-5]+"0.jpg"
    shot_frame1_path = shot_path[:-5]+"1.jpg"
    shot_frame2_path = shot_path[:-5]+"2.jpg"
    assert shot_frame2_path == shot_path

    shot_frame0_pil = Image.open(shot_frame0_path)
    try:
        shot_frame1_pil = Image.open(shot_frame1_path)
    except:
        print("special shot, only 2 frames within the shot") # shot is too short, only 2 frames within the shot
        special_shot+=1
        shot_frame1_pil = Image.open(shot_frame2_path)
    shot_frame2_pil = Image.open(shot_frame2_path)

    shot_frame0_feats = get_image_clip_features(shot_frame0_pil,)
    shot_frame1_feats = get_image_clip_features(shot_frame1_pil,)
    shot_frame2_feats = get_image_clip_features(shot_frame2_pil,)

    feat_dict[shot_frame0_path.replace(args.data_root+"/","")] = shot_frame0_feats
    feat_dict[shot_frame1_path.replace(args.data_root+"/","")] = shot_frame1_feats
    feat_dict[shot_frame2_path.replace(args.data_root+"/","")] = shot_frame2_feats
    # break

print("number of special shots (2 frames): %d"%special_shot)
print("dimension of image embeddings:",shot_frame2_feats.shape)


args.model_version=args.model_version.replace("/","-")
try:
    save_dict_path = SAVE_FOLDER + "/{}_{}.pkl".format(args.model_version, args.phase)
    with open(save_dict_path,"wb") as handle:
        pkl.dump(feat_dict,handle,protocol=pkl.HIGHEST_PROTOCOL)
except:
    save_dict_path = SAVE_FOLDER+"/{}_{}.pkl".format(basename(args.model_version),args.phase)
    with open(save_dict_path,"wb") as handle:
        pkl.dump(feat_dict,handle,protocol=pkl.HIGHEST_PROTOCOL)
