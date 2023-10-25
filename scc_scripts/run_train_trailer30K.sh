DATASET_NAME=trailer30K
DATA_DIR=data/${DATASET_NAME}/MovieCLIP_features
VISUAL_FOLDER=data/${DATASET_NAME}/MovieCLIP_features/Visual_features
VISUAL_FEATURE_VERSION=ViT-L-14@336px
AUDIO_FEATURE_FILE=data/${DATASET_NAME}/MovieCLIP_features/Audio_features/PANNs_embeddings_all.pkl
TEXT_TOKEN_FILE=data/${DATASET_NAME}/MovieCLIP_features/Text_features/whisper_medium_asr_output.json
TEXT_FEATURE_FILE=data/${DATASET_NAME}/MovieCLIP_features/Text_features/whisper_medium_asr_output_ViT-L-14@336px.pkl


export PYTHONPATH=$PWD

###### visual-only model
python train_visual_audio_text.py\
  --model_name=trailer30K_visual\
  --data_dir=${DATA_DIR}\
  --visual_feature_dir=${VISUAL_FOLDER}\
  --visual_feature_version=${VISUAL_FEATURE_VERSION}\
  --audio_feature_file=${AUDIO_FEATURE_FILE}\
  --text_token_file=${TEXT_TOKEN_FILE}\
  --text_feature_file=${TEXT_FEATURE_FILE}\
  --num_epoch=10\
  --include_test=True\
  --include_modality_loss=False\
  --trainable_composition=False\
  --alpha=1.0\
  --beta=0.0

###### audio-only model
python train_visual_audio_text.py\
  --model_name=trailer30K_audio\
  --data_dir=${DATA_DIR}\
  --visual_feature_dir=${VISUAL_FOLDER}\
  --visual_feature_version=${VISUAL_FEATURE_VERSION}\
  --audio_feature_file=${AUDIO_FEATURE_FILE}\
  --text_token_file=${TEXT_TOKEN_FILE}\
  --text_feature_file=${TEXT_FEATURE_FILE}\
  --num_epoch=10\
  --include_test=True\
  --include_modality_loss=False\
  --trainable_composition=False\
  --alpha=0.0\
  --beta=1.0

###### text-only model
python train_visual_audio_text.py\
  --model_name=trailer30K_text\
  --data_dir=${DATA_DIR}\
  --visual_feature_dir=${VISUAL_FOLDER}\
  --visual_feature_version=${VISUAL_FEATURE_VERSION}\
  --audio_feature_file=${AUDIO_FEATURE_FILE}\
  --text_token_file=${TEXT_TOKEN_FILE}\
  --text_feature_file=${TEXT_FEATURE_FILE}\
  --num_epoch=10\
  --include_test=True\
  --include_modality_loss=False\
  --trainable_composition=False\
  --alpha=0.0\
  --beta=0.0

# visual+audio (including separate modality loss)
python train_visual_audio_text.py\
  --model_name=trailer30K_visual_audio_mloss\
  --data_dir=${DATA_DIR}\
  --visual_feature_dir=${VISUAL_FOLDER}\
  --visual_feature_version=${VISUAL_FEATURE_VERSION}\
  --audio_feature_file=${AUDIO_FEATURE_FILE}\
  --text_token_file=${TEXT_TOKEN_FILE}\
  --text_feature_file=${TEXT_FEATURE_FILE}\
  --num_epoch=20\
  --include_test=True\
  --include_modality_loss=True\
  --trainable_composition=False\
  --alpha=0.6\
  --beta=0.4

# visual+audio+text
python train_visual_audio_text.py\
  --model_name=trailer30K_visual_audio_text_mloss\
  --data_dir=${DATA_DIR}\
  --visual_feature_dir=${VISUAL_FOLDER}\
  --visual_feature_version=${VISUAL_FEATURE_VERSION}\
  --audio_feature_file=${AUDIO_FEATURE_FILE}\
  --text_token_file=${TEXT_TOKEN_FILE}\
  --text_feature_file=${TEXT_FEATURE_FILE}\
  --num_epoch=20\
  --include_test=True\
  --include_modality_loss=True\
  --trainable_composition=False\
  --alpha=0.6\
  --beta=0.3

python train_visual_audio_text.py\
  --model_name=trailer30K_visual_audio_text_mloss_flexible\
  --data_dir=${DATA_DIR}\
  --visual_feature_dir=${VISUAL_FOLDER}\
  --visual_feature_version=${VISUAL_FEATURE_VERSION}\
  --audio_feature_file=${AUDIO_FEATURE_FILE}\
  --text_token_file=${TEXT_TOKEN_FILE}\
  --text_feature_file=${TEXT_FEATURE_FILE}\
  --num_epoch=20\
  --include_test=True\
  --include_modality_loss=True\
  --trainable_composition=True\
  --alpha=0.6\
  --beta=0.3