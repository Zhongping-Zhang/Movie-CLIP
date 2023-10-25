import os
from os.path import basename, dirname, join
from PIL import Image
import numpy as np
from sklearn.metrics import average_precision_score
import json
import pickle

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.to(device)
clip_model.eval()


class trailer_shot_dataloader_raw():
    def __init__(self,
                 file_path: str = 'data/trailer30K/key_frame_val.txt',
                 num_classes = 21, 
                 shuffle = False,
                 ):
        self.file_path = file_path
        self.MOVIENET_ROOT = dirname(file_path)
        self.num_classes = num_classes
        
        with open(file_path,'r') as f:
            self.data = f.readlines()
        
        if shuffle:
            np.random.shuffle(self.data)
        
    def __getitem__(self, index: int):
        eles = self.data[index].split("\t")
        
        movie_id = eles[0]
        
        shot_path = eles[1]
        key_frame0_path = shot_path[:-5]+"0.jpg"
        key_frame1_path = shot_path[:-5]+"1.jpg"
        key_frame2_path = shot_path[:-5]+"2.jpg"
        assert key_frame2_path == shot_path

        labels = np.array([int(ele) for ele in eles[2].split(" ")])
        
        labels = torch.LongTensor(labels)
        labels_onehot = nn.functional.one_hot(labels, num_classes=self.num_classes)
        labels_onehot = labels_onehot.sum(dim=0).float()
        
        key_frame0 = Image.open(join(self.MOVIENET_ROOT,key_frame0_path))
        try:
            key_frame1 = Image.open(join(self.MOVIENET_ROOT,key_frame1_path))
        except:
            # print("special shot") # shot is too short, only 2 frames within the shot
            # special_shot+=1
            key_frame1 = Image.open(join(self.MOVIENET_ROOT,key_frame2_path))
        
        
        key_frame2 = Image.open(join(self.MOVIENET_ROOT,key_frame2_path))
        
        key_frame0_feats = clip_model.encode_image(clip_preprocess(key_frame0).unsqueeze(0).to(device))
        key_frame1_feats = clip_model.encode_image(clip_preprocess(key_frame1).unsqueeze(0).to(device))
        key_frame2_feats = clip_model.encode_image(clip_preprocess(key_frame2).unsqueeze(0).to(device))

        return (key_frame0_feats, key_frame1_feats, key_frame2_feats, movie_id), labels_onehot
        
    def __len__(self):
        return len(self.data)   

class trailer_shot_dataloader():
    def __init__(self,
                 file_path: str = 'data/trailer30K/key_frame_val.txt',
                 visual_feat_folder: str = "data/trailer30K/CLIP_features/",
                 num_classes = 21, 
                 shuffle = False,
                 ):
        self.file_path = file_path
        self.MOVIENET_ROOT = dirname(file_path)
        self.num_classes = num_classes
        self.visual_feat_folder = visual_feat_folder
        if "val" in file_path:
            phase = 'val'
        elif "train" in file_path:
            phase='train'
        else:
            phase='test'
            
        with open(file_path,'r') as f:
            self.data = f.readlines()
        
        
        img_feature_path = "imgfeat_32_{}.pkl".format(phase)
        print('load clip features from: ',self.visual_feat_folder+img_feature_path)
        with open(os.path.join(self.visual_feat_folder, img_feature_path),'rb') as handle:
            self.feat_dict = pickle.load(handle)
            
        if shuffle:
            np.random.shuffle(self.data)
        
    def __getitem__(self, index: int):
        eles = self.data[index].split("\t")
        
        movie_id = eles[0]
        
        shot_path = eles[1]
        key_frame0_path = shot_path[:-5]+"0.jpg"
        key_frame1_path = shot_path[:-5]+"1.jpg"
        key_frame2_path = shot_path[:-5]+"2.jpg"
        assert key_frame2_path == shot_path

        labels = np.array([int(ele) for ele in eles[2].split(" ")])
        labels = torch.LongTensor(labels)
        labels_onehot = nn.functional.one_hot(labels, num_classes=self.num_classes)
        labels_onehot = labels_onehot.sum(dim=0).float()
        
        # key_frame0 = Image.open(join(self.MOVIENET_ROOT,key_frame0_path))
        # try:
        #     key_frame1 = Image.open(join(self.MOVIENET_ROOT,key_frame1_path))
        # except:
        #     key_frame1 = Image.open(join(self.MOVIENET_ROOT,key_frame2_path))
        # key_frame2 = Image.open(join(self.MOVIENET_ROOT,key_frame2_path))
        
        # key_frame0_feats = clip_model.encode_image(clip_preprocess(key_frame0).unsqueeze(0).to(device))
        # key_frame1_feats = clip_model.encode_image(clip_preprocess(key_frame1).unsqueeze(0).to(device))
        # key_frame2_feats = clip_model.encode_image(clip_preprocess(key_frame2).unsqueeze(0).to(device))
        
        key_frame0_feats = self.feat_dict[key_frame0_path]
        key_frame1_feats = self.feat_dict[key_frame1_path]
        key_frame2_feats = self.feat_dict[key_frame2_path]
            
        return (key_frame0_feats, key_frame1_feats, key_frame2_feats, movie_id), labels_onehot
        
    def __len__(self):
        return len(self.data)   
    
    

def DensewithBN(in_fea, out_fea, normalize=True, dropout=False):
    layers=[nn.Linear(in_fea, out_fea)]
    if normalize==True:
        layers.append(nn.BatchNorm1d(num_features = out_fea))
    layers.append(nn.ReLU())
    if dropout:
        layers.append(nn.Dropout(p=0.5))
    return layers

class trailer_shot_clipmodel(nn.Module):
    def __init__(self,
                 img_feat_dim: int = 512, # dimension of clip features
                 hidden_dim: int = 512,
                 num_category: int = 21,
                 dropout: bool=False,
                 ):
        super(trailer_shot_clipmodel, self).__init__()
        self.imgdense = nn.Sequential(*DensewithBN(img_feat_dim, hidden_dim, dropout=dropout),
                                      nn.Linear(hidden_dim,num_category),
                                      )
        
    def forward(self, key_frame0_feats, key_frame1_feats, key_frame2_feats):        
        key_frame_avg = torch.mean(torch.stack([key_frame0_feats,key_frame1_feats,key_frame2_feats]),dim=0)
        output = self.imgdense(key_frame_avg)
        return output
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(model, dataloader, optimizer, epoch):
    model.train()
    for batch_idx,(inputs,labels) in enumerate(dataloader):
        k1_clip = inputs[0][:,0,:].type(FloatTensor)
        k2_clip = inputs[1][:,0,:].type(FloatTensor)
        k3_clip = inputs[2][:,0,:].type(FloatTensor)
        labels = labels.type(LongTensor) # convert to gpu computation
        
        optimizer.zero_grad() # optimizer.zero_grad() IMPORTANT!
        output = model(k1_clip,k2_clip,k3_clip)
        
        loss = F.binary_cross_entropy_with_logits(output, labels.float())
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.process_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(labels), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def val(model, dataloader):
    return_predictions = []
    return_labels = []
    with torch.no_grad():
        model.eval()#model.eval() fix the BN and Dropout
        val_loss = 0
        for batch_idx,(inputs,labels) in enumerate(dataloader):
            k1_clip = inputs[0][:,0,:].type(FloatTensor)
            k2_clip = inputs[1][:,0,:].type(FloatTensor)
            k3_clip = inputs[2][:,0,:].type(FloatTensor)
            labels = labels.type(LongTensor) # convert to gpu computation
            
            output = model(k1_clip, k2_clip, k3_clip)
        
            val_loss += BCE_criterion(output, labels.float())  #outputs – (N,C); target – (N)
            predicted = output.data
            predicted_np = predicted.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            return_predictions.append(predicted_np)
            return_labels.append(labels_np)
        
        predicted_np, labels_np = np.concatenate(return_predictions), np.concatenate(return_labels)
        
        mAP = average_precision_score(labels_np, sigmoid(predicted_np) )
        val_loss /= len(val_loader)

        print('\nValidation set: Average loss: {:.4f}, mAP: {:.4f} \n'
              .format(val_loss, mAP) )
        
    return labels_np, output.data, val_loss, mAP

            

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='shot_genre_detector', help='model name')
    parser.add_argument('--file_path', type=str, default="data/trailer30K/shotinfo_train.txt", help='path to save data files')
    parser.add_argument('--epoch_num', type=int, default=10, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    
    parser.add_argument('--img_dim', type=int, default = 512, help='dimensionality of image feature')
    parser.add_argument('--hidden_dim', type=int, default = 512, help='dimensionality of hidden feature')
    parser.add_argument('--num_categories',type=int,default = 21, help='The number of categories')
    parser.add_argument('--dropout',type=bool,default = True, help='use dropout or not')
    
    parser.add_argument('--include_test', type=bool, default = True, help='do test or not')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--save_interval',type=int,default=2,help='the interval between saved epochs')
    parser.add_argument('--process_interval',type=int,default=100,help='the interval between process print')
    
    args = parser.parse_args()
    print(args)
    
    BCE_criterion = nn.BCEWithLogitsLoss()
    os.makedirs('logs/'+args.model_name, exist_ok=True)

    traindata = trailer_shot_dataloader(
                             file_path=args.file_path, 
                             num_classes=args.num_categories,
                             shuffle=True,
                             )

    valdata = trailer_shot_dataloader(
                            file_path = join(dirname(args.file_path),"shotinfo_val.txt"),
                            num_classes=args.num_categories,
                            shuffle=False,
                            )
    
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    val_loader = torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)
    
    
    model = trailer_shot_clipmodel(img_feat_dim=args.img_dim, 
                                   hidden_dim=args.hidden_dim, 
                                   num_category=args.num_categories, 
                                   dropout = args.dropout,
                                   ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=0, verbose=True)
    
    best_macro_mAP = 0

    f_logger = open("logs/"+args.model_name+"/logger_info.txt",'w')
    for epoch_org in range(args.epoch_num):
        epoch = epoch_org+1
        train(model, train_loader, optimizer, epoch)
        _,_,val_loss,macro_mAP = val(model, val_loader)
        scheduler.step(macro_mAP)
        
        f_logger.write("epoch-{}: val: {:.4f}; mAP: {:.4f} \n".format(epoch, val_loss, macro_mAP))
        if macro_mAP>best_macro_mAP:
            best_macro_mAP=macro_mAP
            torch.save(model, "logs/"+args.model_name+"/epoch-best.pkl")
            best_epoch = epoch
        if epoch%args.save_interval == args.save_interval-1:
            print('saving the %d epoch' %(epoch))
            torch.save(model, "logs/"+args.model_name+"/epoch-%d.pkl" %(epoch))
            
    f_logger.write("best epoch num: %d"%best_epoch)
    f_logger.close()
    
    
    results = vars(args)
    results.update({'best_epoch_mAP': best_macro_mAP, 'best_epoch': best_epoch})
    with open(os.path.join("logs",args.model_name,"results.json"), 'w') as f:
        json.dump(results,f)

    
