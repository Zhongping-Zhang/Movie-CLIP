# Movie Genre Classification by Language Augmentation and Shot Sampling
**Movie-CLIP** is a video-based movie genre classification model. This repository contains the PyTorch implementation for our [paper](https://arxiv.org/abs/2203.13281).
If you find this code useful in your research, please consider citing:

    @InProceedings{zhangMovie2024,
         author={Zhongping Zhang and Yiwen Gu and Bryan A. Plummer and Xin Miao and Jiayi Liu and Huayan Wang},
         title={Movie Genre Classification by Language Augmentation and Shot Sampling},
         booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV)},
         year={2024}}

## Set up environment
Libraries to train Movie-CLIP can be installed by:
```sh
pip install -r requirements.txt
```

## Datasets:
We performed our experiments on two datasets, [MovieNet](http://movienet.site/) and [CondensedMovies](https://www.robots.ox.ac.uk/~vgg/data/condensed-movies/). 
We provided the data we employed for model training and evaluation through the following links. 


| Datasets              | Google Drive Link                                                                                     |
|-----------------------|-------------------------------------------------------------------------------------------------------|
| MovieNet (Trailer30K) | [Trailer30K Link](https://drive.google.com/drive/folders/13383-assGkSU-KO1sNdsJLdvBCvVV4oG?usp=sharing) |
| CondensedMovies       | [Condensed Movies Link](https://drive.google.com/drive/folders/1RVd_A_JXQtfbVxQ9i9HlP7yphnGIfgq7?usp=sharing)|

If you would like to obtain the original data, we also provide the [code](https://github.com/Zhongping-Zhang/Movie-CLIP/blob/main/download_youtube_videos_captions.py) to collect videos 
using [PyTube](https://github.com/pytube/pytube) and [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/).

Take Trailer30K as an example:

**Step 1** download trailer URLs and meta data
```sh
 wget https://download.openmmlab.com/datasets/movienet/trailer.urls.unique30K.v1.json # version 1.0
 wget https://download.openmmlab.com/datasets/movienet/meta.v1.zip # version 1.0
```

**Step 2** download source videos from YouTube

```sh
python download_youtube_videos_captions.py
```


## Train & Evaluate the Movie-CLIP model
### Train and Evaluate mAP
Run the following script to train and evaluate Movie-CLIP on trailer30K:
```sh
sh scc_scripts/run_train_trailer30K.sh
```

Run the following script to train and evaluate Movie-CLIP on Condensed Movies:
```sh
sh scc_scripts/run_train_condensed_movies.sh
```

**Note**: To smoothly run the scripts, **variables** including *DATA_DIR*, 
*VISUAL_FOLDER*, *VISUAL_FEATURE_VERSION*, *AUDIO_FEATURE_FILE*,
*TEXT_TOKEN_FILE*, *TEXT_FEATURE_FILE* need to be correctly defined. 
