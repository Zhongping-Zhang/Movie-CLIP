import pickle

with open("data/CondensedMovies/processed_features/audio_features_16000/audiofeat_all.pkl", "rb") as f:
    data=pickle.load(f)

print("aaa")
# import whisper
# import argparse
# import os
# import time
# import torchaudio
# import transformers
# from transformers import AutoProcessor, WhisperForConditionalGeneration
# from transformers import pipeline
# import torch
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--phase', default='val', choices=['train', 'test', 'val'])
# parser.add_argument('--data_root', default='data/CondensedMovies/MovieCLIP_features')
# parser.add_argument('--whisper_version', default='large-v2')
# parser.add_argument('--cache_dir', type=str, default="/projectnb/ivc-ml/zpzhang/checkpoints/whisper_cache")
# args = parser.parse_args()
#
#
# os.makedirs(args.cache_dir,exist_ok=True)
#
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# pipe = pipeline(
#   "automatic-speech-recognition",
#   model="openai/whisper-base.en",
#   chunk_length_s=30,
#   device=device,
#   # cache_dir=args.cache_dir,
# )
#
#
# from datasets import load_dataset
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample = ds[0]["audio"]
#
# waveform, sampling_rate = torchaudio.load('data/CondensedMovies/processed_features/sample_audio/0NUDP-gxGyM_16000.wav')
# sample2={
#         'array': waveform.mean(dim=0).numpy(),
#         'sampling_rate': sampling_rate,
# }
#
# prediction = pipe(sample.copy(), batch_size=8)["text"]
# print(prediction)
# # processor = AutoProcessor.from_pretrained("openai/whisper-base.en",cache_dir=args.cache_dir)
# # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en",cache_dir=args.cache_dir)
# # feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base",cache_dir=args.cache_dir)


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