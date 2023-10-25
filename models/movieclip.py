import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def DensewithBN(in_fea, out_fea, normalize=True, dropout=False):
    layers = [nn.Linear(in_fea, out_fea)]
    if normalize == True:
        layers.append(nn.BatchNorm1d(num_features=out_fea))
    layers.append(nn.ReLU())
    if dropout:
        layers.append(nn.Dropout(p=0.5))
    return layers


class trailer_visual(nn.Module):
    def __init__(self,
                 visual_feature_dim: int = 2048,  # resnet50 feature dimension
                 hidden_dim: int = 512,
                 num_category: int = 21,
                 concatenate: str = "mean",
                 dropout: str = True,
                 ):
        super(trailer_visual, self).__init__()
        self.visual_feature_dim=visual_feature_dim
        self.hidden_dim=hidden_dim
        self.num_category=num_category
        self.dropout=dropout
        self.concatenate = concatenate
        if self.concatenate == 'concatenate': visual_feature_dim = visual_feature_dim * 8

        self.visual_dense1 = nn.Sequential(*DensewithBN(visual_feature_dim, hidden_dim, dropout=dropout))
        self.visual_dense2 = nn.Linear(hidden_dim, num_category)

    def forward(self, visual_features):
        if self.concatenate == 'average' or self.concatenate == 'mean':
            concatenate_visual_features = torch.mean(visual_features, dim=1)  # (batch_size, num_shot, feat_dim) ==> (batch_size, feat_dim)
        elif self.concatenate == 'concatenate':
            batch_size = visual_features.shape[0]
            concatenate_visual_features = visual_features.view(batch_size, -1)
        else:
            assert False, "please choose a valid way to concatenate features: [average/mean, concatenate]"

        visual_dense1_output = self.visual_dense1(concatenate_visual_features)
        self.movieclip_visual_embeds = visual_dense1_output.detach()
        visual_output = self.visual_dense2(visual_dense1_output)
        return {"visual_output": visual_output,
                "multimodal_output": visual_output,}

class trailer_audio(nn.Module):
    def __init__(self,audio_feature_dim=2048, hidden_dim=512, num_category=21, dropout=True):
        super(trailer_audio, self).__init__()
        self.audio_dense1 = nn.Sequential(*DensewithBN(audio_feature_dim, hidden_dim, dropout=dropout))
        self.audio_dense2 = nn.Linear(hidden_dim, num_category)

    def forward(self, audio_features):
        audio_dense1_output = self.audio_dense1(audio_features)
        self.movieclip_audio_embeds = audio_dense1_output.detach()
        audio_output = self.audio_dense2(audio_dense1_output)
        return {'audio_output': audio_output,
                'multimodal_output': audio_output,}

class trailer_text(nn.Module):
    def __init__(self,text_feature_dim=768, hidden_dim=512, num_category=21, dropout=True):
        super(trailer_text, self).__init__()
        self.text_dense1 = nn.Sequential(*DensewithBN(text_feature_dim, hidden_dim, dropout=dropout))
        self.text_dense2 = nn.Linear(hidden_dim, num_category)

    def forward(self, text_features):
        text_dense1_output = self.text_dense1(text_features) # (None,768)
        self.movieclip_text_embeds = text_dense1_output.detach()
        text_output = self.text_dense2(text_dense1_output)
        return {'text_output': text_output,
                'multimodal_output': text_output,}


class trailer_visual_audio(trailer_visual):
    def __init__(self, audio_feature_dim=2048, trainable_composition=False, **kwargs):
        super(trailer_visual_audio,self).__init__(**kwargs)
        self.audio_dense1 = nn.Sequential(*DensewithBN(audio_feature_dim,self.hidden_dim,dropout=self.dropout))
        self.audio_dense2 = nn.Linear(self.hidden_dim, self.num_category)

        self.trainable_composition=trainable_composition
        self.alpha = nn.Parameter(torch.Tensor([0.5]))  # initialize alpha as 0.5
        self.beta = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, visual_features, audio_features):
        if self.concatenate == 'average' or self.concatenate == 'mean':
            concatenate_visual_features = torch.mean(visual_features, dim=1)  # (batch_size, num_shot, feat_dim) ==> (batch_size, feat_dim)
        elif self.concatenate == 'concatenate':
            batch_size = visual_features.shape[0]
            concatenate_visual_features = visual_features.view(batch_size, -1)
        else:
            assert False, "please choose a valid way to concatenate features: [average/mean, concatenate]"

        visual_dense1_output = self.visual_dense1(concatenate_visual_features) # concatenate_visual_features: (batch_size,768)
        self.movieclip_visual_embeds = visual_dense1_output.detach()
        visual_output = self.visual_dense2(visual_dense1_output)

        audio_dense1_output = self.audio_dense1(audio_features)
        self.movieclip_audio_embeds = audio_dense1_output.detach()
        audio_output = self.audio_dense2(audio_dense1_output)

        if self.trainable_composition is True:
            multimodal_output = visual_output*self.alpha+audio_output*(1-self.alpha)
        else:
            multimodal_output = (visual_output+audio_output)/2


        return {"visual_output": visual_output,
                "audio_output": audio_output,
                "multimodal_output": multimodal_output,
                }

class trailer_visual_audio_text(trailer_visual_audio):
    def __init__(self, text_feature_dim=768, alpha=0.5, beta=0.5, **kwargs):
        super(trailer_visual_audio_text,self).__init__(**kwargs)
        self.text_dense1 = nn.Sequential(*DensewithBN(text_feature_dim,self.hidden_dim,dropout=self.dropout))
        self.text_dense2 = nn.Linear(self.hidden_dim, self.num_category)

        if self.trainable_composition is True:
            self.alpha = nn.Parameter(torch.Tensor([alpha]))
            self.beta = nn.Parameter(torch.Tensor([beta]))
        else:
            self.alpha=nn.Parameter(torch.Tensor([alpha]),requires_grad=False)
            self.beta=nn.Parameter(torch.Tensor([beta]),requires_grad=False)

    def forward(self, visual_features, audio_features, text_features):
        if self.concatenate == 'average' or self.concatenate == 'mean':
            concatenate_visual_features = torch.mean(visual_features, dim=1)  # (batch_size, num_shot, feat_dim) ==> (batch_size, feat_dim)
        elif self.concatenate == 'concatenate':
            batch_size = visual_features.shape[0]
            concatenate_visual_features = visual_features.view(batch_size, -1)
        else:
            assert False, "please choose a valid way to concatenate features: [average/mean, concatenate]"

        visual_dense1_output = self.visual_dense1(concatenate_visual_features) # concatenate_visual_features: (batch_size,768)
        self.movieclip_visual_embeds = visual_dense1_output.detach()
        visual_output = self.visual_dense2(visual_dense1_output)

        audio_dense1_output = self.audio_dense1(audio_features)
        self.movieclip_audio_embeds = audio_dense1_output.detach()
        audio_output = self.audio_dense2(audio_dense1_output)

        text_dense1_output = self.text_dense1(text_features)
        self.movieclip_text_embeds = text_dense1_output.detach()
        text_output = self.text_dense2(text_dense1_output)

        multimodal_output = visual_output*self.alpha+audio_output*self.beta+text_output*(1-self.alpha-self.beta)

        return {"visual_output": visual_output,
                "audio_output": audio_output,
                "text_output": text_output,
                "multimodal_output": multimodal_output,
                }



if __name__ == "__main__":
    import numpy as np
    import json

    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 2


    visual_features = torch.randn(batch_size, 8, 512).type(FloatTensor) # (None,8,512)
    audio_features = torch.randn(batch_size, 2048).type(FloatTensor)

    labels = np.ones((batch_size, 21))
    labels = torch.from_numpy(labels).type(LongTensor)

    text = torch.randint(0, 10000, (batch_size, 256)).type(LongTensor)
    mask = torch.ones_like(text).bool().type(LongTensor)
    #    model = trailer_shot_baseline(concatenate='concatenate', finetune=False).to(device)

    model = trailer_visual_audio(visual_feature_dim=512, audio_feature_dim=2048).to(
        device)

    output = model(visual_features, audio_features)
    # output = model(feats)
    output_np = output.cpu().detach().numpy()

# =============================================================================
#     # model = trailer_baseline(concatenate='concatenate').to(device)
#     model = trailer_baseline(img_feat_dim=2048,
#                             hidden_dim=512,
#                             num_category=21,
#                             concatenate="mean",
#                             ).to(device)
#     output = model(feats)
# =============================================================================


#    print(output.shape)
#    print(output_np)

# class trailer_visual_audio(trailer_visual):
#     def __init__(self, audio_feature_dim=2048, reshape_audio_dim=False, **kwargs):
#         super(trailer_visual_audio, self).__init__(**kwargs)
#         self.reshape_audio_dim=reshape_audio_dim
#         print("aaaaaaaaaaaa")
#         if self.reshape_audio_dim is True:
#             print("bbbbbbbbbbbbbbb")
#             self.audio_linear = nn.Linear(audio_feature_dim, self.hidden_dim)
#             audio_feature_dim=self.hidden_dim # (2048->512)
#             print('audio_feature_dim',audio_feature_dim)
#         self.multi_dense = nn.Sequential(*DensewithBN(self.visual_feature_dim+audio_feature_dim, self.hidden_dim, dropout=self.dropout),
#                                          nn.Linear(self.hidden_dim, self.num_category))
#
#     def forward(self, visual_features, audio_features):
#         if self.concatenate == 'average' or self.concatenate == 'mean':
#             concatenate_visual_features = torch.mean(visual_features, dim=1)  # (batch_size, num_shot, feat_dim) ==> (batch_size, feat_dim)
#         elif self.concatenate == 'concatenate':
#             batch_size = visual_features.shape[0]
#             concatenate_visual_features = visual_features.view(batch_size, -1)
#         else:
#             assert False, "please choose a valid way to concatenate features: [average/mean, concatenate]"
#
#         if self.reshape_audio_dim is True:
#             audio_features = self.audio_linear(audio_features)
#         multimodal_features = torch.cat([concatenate_visual_features, audio_features], dim=1)
#         output = self.multi_dense(multimodal_features)
#         return output


