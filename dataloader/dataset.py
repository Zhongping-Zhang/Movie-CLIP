import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class trailer_multimodal_features(Dataset):
    def __init__(self,
                 phase: str='train',
                 data_dir: str='data/CondensedMovies/MovieCLIP_features',
                 visual_feature_dir: str='data/CondensedMovies/MovieCLIP_features/Visual_features',
                 visual_feature_file: str=None,
                 audio_feature_file: str=None,
                 text_token_file: str=None,
                 text_feature_file: str=None,
                 num_classes: int=21,
                 ):
        self.phase = phase
        self.data_dir = data_dir
        self.visual_feature_dir = visual_feature_dir
        self.audio_feature_file = audio_feature_file
        self.text_token_file = text_token_file
        self.text_feature_file = text_feature_file
        self.num_classes = num_classes

        # loading dataset samples
        with open(os.path.join(self.data_dir, "data_{}.json".format(phase)), 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.samples = list(self.data.values())


        if self.visual_feature_dir is not None:
            if visual_feature_file is None:
                visual_feature_file="ViT-B-32_{}.pkl".format(phase)
            print('load visual features from:', os.path.join(self.visual_feature_dir, visual_feature_file))
            with open(os.path.join(self.visual_feature_dir, visual_feature_file) ,'rb') as handle:
                self.visual_feature_dict = pickle.load(handle)

        if self.audio_feature_file is not None:
            print('load audio features from:', self.audio_feature_file)
            with open(audio_feature_file, 'rb') as handle:
                self.audio_feature_dict = pickle.load(handle)

        if self.text_token_file is not None:
            print('load text tokens from:', self.text_token_file)
            with open(text_token_file,'r') as handle:
                self.text_token_dict = json.load(handle)
            if self.text_feature_file is not None:
                print('load text features from:', self.text_feature_file)
                with open(text_feature_file,"rb") as handle:
                    self.text_feature_dict = pickle.load(handle)

    def __getitem__(self, index: int):
        movie_id = self.keys[index]
        sample = self.samples[index]
        labels = np.array([int(ele) for ele in sample['label'].split(" ")])

        labels = torch.LongTensor(labels) # shape: (1)
        labels_onehot = nn.functional.one_hot(labels, num_classes=self.num_classes) # (postive_labels,num_classes)
        labels_onehot = labels_onehot.sum(dim=0).float() # (num_classes)

        return_dict = {
            "movie_id": movie_id,
            "label_onehot": labels_onehot,
        }

        if self.visual_feature_dir is not None:
            visual_feature_list = []
            for key, value in sample.items():
                if key=='label': continue
                shot_frame0_path = value[0].replace(self.data_dir+"/","",1)
                shot_frame1_path = value[1].replace(self.data_dir+"/","",1)
                shot_frame2_path = value[2].replace(self.data_dir+"/","",1)

                shot_frame0_feature = self.visual_feature_dict[shot_frame0_path] # (1,512)
                shot_frame1_feature = self.visual_feature_dict[shot_frame1_path] # (1,512)
                shot_frame2_feature = self.visual_feature_dict[shot_frame2_path] # (1,512)

                visual_feature_list.append(np.mean([shot_frame0_feature, shot_frame1_feature, shot_frame2_feature], axis=0)) # (1,512)

            assert len(visual_feature_list)==8
            visual_features = np.concatenate(visual_feature_list ,axis=0) # (8,512)-> (num_of_shot, feat_embed_size)
            return_dict["visual_feature"] = visual_features

        if self.audio_feature_file is not None:
            audio_features = self.audio_feature_dict[movie_id+".wav"]["embedding"]
            return_dict["audio_feature"] = audio_features # (2048,), PANNs feature

        if self.text_token_file is not None:
            text_tokens = self.text_token_dict[movie_id+".wav"]
            return_dict["text_tokens"] = text_tokens
            if self.text_feature_file is not None:
                text_features = self.text_feature_dict[movie_id+".wav"]
                return_dict["text_feature"] = text_features

        return return_dict

    def __len__(self):
        return len(self.data)


# class trailer_visual_audio_features(trailer_visual_features):
#     def __init__(self, , **kwargs):
#         super(trailer_visual_audio_features, self).__init__(**kwargs)




if __name__=="__main__":
    batch_size = 2
    shuffle = False

    audio_feature_file=None
    # audio_feature_file="data/CondensedMovies/MovieCLIP_features/Audio_features/PANNs_embeddings_all.pkl"
    text_feature_file=None
    text_feature_file="data/CondensedMovies/MovieCLIP_features/Text_features/whisper_medium_asr_output.json"
    dataset = trailer_multimodal_features(phase='val',
                                          audio_feature_file=audio_feature_file,
                                          text_feature_file=text_feature_file,
                                          )


    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             )

    idx = 0
    for batch_idx, sample in enumerate(dataloader):
        pass
        # print(inputs[0].shape, inputs[-1], labels)
        # print(inputs[1].shape)
        # break




