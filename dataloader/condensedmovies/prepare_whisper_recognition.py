import whisper
import argparse
import os,glob,json
from os.path import join, basename, dirname
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='val', choices=['train', 'test', 'val'])
parser.add_argument('--data_root', default='data/CondensedMovies/MovieCLIP_features/Audio_features/audio_original_sr16000')
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

        # break

# result = model.transcribe("data/CondensedMovies/processed_features/sample_audio/0NUDP-gxGyM_16000.wav")
# print(result["text"])




# ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
# sample = next(iter(ds))

# processor = AutoProcessor.from_pretrained("openai/whisper-base.en",cache_dir=args.cache_dir)
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en",cache_dir=args.cache_dir)
# feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base",cache_dir=args.cache_dir)

# waveform, sampling_rate = torchaudio.load('data/CondensedMovies/processed_features/sample_audio/0NUDP-gxGyM_16000.wav')
# sample={
#     'audio':{
#         'array': waveform.mean(dim=0),
#         'sampling_rate': sampling_rate,},
# }
#
# start_time = time.time()
#
#
# inputs = processor(sample["audio"]["array"],sampling_rate=sample['audio']['sampling_rate'],return_tensors="pt")
# input_features = inputs.input_features
# predicted_ids = model.generate(input_features)
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
#
# # generated_ids = model(input_features=input_features, decoder_input_ids=decoder_input_ids)
#
# transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(transcription)


# inputs = feature_extractor(
#     sample['audio']["array"], sampling_rate=sample['audio']["sampling_rate"], return_tensors="pt"
# )
# input_features = inputs.input_features



# end_time = time.time()
# run_time = end_time-start_time
# print("程序运行时间：", run_time, "秒")
# print(inputs)
# print('features',input_features.shape)



"""
0NUDP-gxGyM_16000:
Well, perhaps you'll have better luck opening that. Margot 58, a common fitage. How fascinating that an astronomical 
event can affect quality. How is it going with our case you refuse to take? Of the wine. Oh, I've hit a dead end, literally.
 A found your man is buried in blackwoods tomb. If you still need him. Oh dear. I hope my client doesn't come looking for a refund.
  He's a professor, isn't he? I couldn't see his face, but I spotted a bit of chalk on his lap. I've never known a professor to 
  carry a gun and on such a clever contraption. I patch. Nice touch. So, case closed. Which makes this a social visit? 
  No, it's a urine over your head Irene, isn't it? Whoever killed Rieden was covering their tracks, which makes your next 
  loose end to be snipped. Let it breathe. I've never been in over my head. Leave now. Disappear. You'll get it that. We'll stay and volunteer for protective custody. If I'm in danger, so are you. Come with me. What if we trusted each other? Hmm? You're not listening. I'm taking you to either the railway station or the police station. Hmm. Sir. Oh, which is it to be? Yes. You do that. Which will it be? I told you to let it breathe. It takes the comet. Why couldn't you just come away with me? No. Never. Thank you.

0NUDP-gxGyM_8000:

"""