export PYTHONPATH=$PWD


python dataloader/movienet/prepare_genre_statistics.py
python dataloader/movienet/prepare_get_key_frame_info.py
python dataloader/movienet/prepare_split_testset.py
python dataloader/movienet/prepare_get_visual_features.py --phase=val --model_version=ViT-L/14@336px

module load ffmpeg # load ffmpeg from scc
python dataloader/movienet/prepare_whisper_recognition.py --whisper_version=medium.en --start_idx=0 --end_idx=5

python dataloader/movienet/prepare_get_text_features.py





