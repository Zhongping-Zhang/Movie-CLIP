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



