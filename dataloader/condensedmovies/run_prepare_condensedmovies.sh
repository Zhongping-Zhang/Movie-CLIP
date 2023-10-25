export PTYHONPATH=$PWD

python dataloader/condensedmovies/prepare_imdb2genre.py
python dataloader/condensedmovies/prepare_genre_statistics.py
python dataloader/condensedmovies/prepare_split_testset.py --save_genre_info=True

python dataloader/condensedmovies/prepare_get_visual_features.py --phase=test --model_version=EVA02-CLIP-B-16
module load ffmpeg
python dataloader/condensedmovies/prepare_whisper_recognition.py --whisper_version=medium.en --start_idx=10000 --end_idx=15000
python dataloader/condensedmovies/prepare_get_text_features.py




# vocab/imdb2clip.json:


# prepare_dataloader_dict.py: convert .txt file to .json.file


