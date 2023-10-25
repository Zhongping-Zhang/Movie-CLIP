import whisper
import argparse
import os,glob,json
from os.path import join, basename, dirname
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='val', choices=['train', 'test', 'val'])
parser.add_argument('--data_root', default='data/trailer30K/MovieCLIP_features/Audio_features/trailer30K_audio_sr16000')
parser.add_argument('--whisper_version', default='medium.en')
parser.add_argument('--cache_dir', type=str, default="/projectnb/ivc-ml/zpzhang/checkpoints/whisper_cache")
parser.add_argument('--start_idx',type=int, default=0)
parser.add_argument('--end_idx',type=int,default=5000)
args = parser.parse_args()

save_folder = dirname(args.data_root)


os.makedirs(args.cache_dir,exist_ok=True)
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(args.whisper_version, download_root=args.cache_dir).to(device)

audio_list = sorted(glob.glob(args.data_root+"/*.wav"))


asr_dict = {}

audio_list = audio_list[args.start_idx:args.end_idx]
for audio_path in tqdm(audio_list):
    result = model.transcribe(audio_path) # result keys: language, segments, text
    # print(result["text"])
    asr_dict[os.path.basename(audio_path)] = result["text"]

with open(os.path.join(save_folder, "%s_%d_%d.json"%(args.whisper_version,args.start_idx,args.end_idx)), "w") as fj:
    json.dump(asr_dict, fj)
