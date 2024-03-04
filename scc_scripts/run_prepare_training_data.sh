
export PYTHONPATH=$PWD

######### Extract EVA02-CLIP Features ############
python dataloader/condensedmovies/prepare_get_visual_features.py \
  --clip_version=eva_clip \
  --model_version=EVA02-CLIP-bigE-14-plus \
  --phase=train \


######### Extract CLIP Features ############

python dataloader/condensedmovies/prepare_get_visual_features.py \
  --clip_version=clip \
  --model_version=ViT-L/14@336px \
  --phase=val \
  --cache_dir=/projectnb/ivc-ml/zpzhang/checkpoints/clip_cache

python dataloader/condensedmovies/prepare_get_visual_features.py \
  --clip_version=clip \
  --model_version=ViT-L/14@336px \
  --phase=test \
  --cache_dir=/projectnb/ivc-ml/zpzhang/checkpoints/clip_cache

python dataloader/condensedmovies/prepare_get_visual_features.py \
  --clip_version=clip \
  --model_version=ViT-L/14@336px \
  --phase=train \
  --cache_dir=/projectnb/ivc-ml/zpzhang/checkpoints/clip_cache

