import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os
import json
from sklearn.metrics import average_precision_score
from dataloader.dataset import trailer_multimodal_features
from models.movieclip import trailer_visual

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        movie_id, label, visual_feature = sample['movie_id'],sample['label_onehot'],sample['visual_feature']
        visual_feature = visual_feature.to(device).type(FloatTensor)
        label = label.type(LongTensor)

        optimizer.zero_grad()
        output = model(visual_feature)["multimodal_output"]

        loss = F.binary_cross_entropy_with_logits(output, label.float())
        loss.backward()
        optimizer.step()

        if batch_idx % args.process_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(label), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def val():
    global val_loader

    return_predictions = []
    return_labels = []
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, sample in enumerate(val_loader):
            movie_id, label, visual_feature = sample['movie_id'],sample['label_onehot'],sample['visual_feature']
            visual_feature = visual_feature.to(device).type(FloatTensor)
            label = label.type(LongTensor)  # convert to gpu computation

            output = model(visual_feature)["multimodal_output"]

            val_loss += BCE_criterion(output, label.float())  # outputs – (N,C); target – (N)
            predicted = output.data
            predicted_np = predicted.cpu().numpy()
            label_np = label.cpu().numpy()

            return_predictions.append(predicted_np)
            return_labels.append(label_np)

        predictions_np, labels_np = np.concatenate(return_predictions), np.concatenate(return_labels)

        mAP = average_precision_score(labels_np, sigmoid(predictions_np))
        val_loss /= len(val_loader)

        print('\nValidation set: Average loss: {:.4f}, mAP: {:.4f} \n'
              .format(val_loss, mAP))
    return labels_np, output.data, val_loss, mAP


def main(args):
    global BCE_criterion, train_loader, val_loader, model, optimizer
    BCE_criterion = nn.BCEWithLogitsLoss()

    log_name = os.path.join("logs", args.model_name+"_"+args.visual_feature_version)
    os.makedirs(log_name, exist_ok=True)

    """ Step1: Configure dataset & model & optimizer """
    traindata = trailer_multimodal_features(phase='train',
                                        data_dir=args.data_dir,
                                        visual_feature_dir=args.visual_feature_dir,
                                        visual_feature_file=args.visual_feature_version+"_train.pkl",
                                        audio_feature_file=args.audio_feature_file,
                                        num_classes=args.num_categories,
                                        )
    valdata = trailer_multimodal_features(phase='val',
                                        data_dir=args.data_dir,
                                        visual_feature_dir=args.visual_feature_dir,
                                        visual_feature_file=args.visual_feature_version+"_val.pkl",
                                        audio_feature_file=args.audio_feature_file,
                                        num_classes=args.num_categories,
                                        )

    train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    val_loader = torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)


    test_sample = traindata[0]
    visual_feature_dim = test_sample['visual_feature'].shape[1]

    model = trailer_visual(visual_feature_dim=visual_feature_dim,
                           hidden_dim=args.hidden_dim,
                           num_category=args.num_categories,
                           concatenate=args.concatenate,
                           dropout=args.dropout,
                           ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=0, verbose=True)

    best_macro_mAP = 0

    f_logger = open(log_name + "/logger_info.txt", 'w')
    for epoch_org in range(args.num_epoch):
        epoch = epoch_org + 1
        train(epoch)
        _, _, val_loss, macro_mAP = val()
        scheduler.step(macro_mAP)

        f_logger.write("epoch-{}: val: {:.4f}; mAP: {:.4f} \n".format(epoch, val_loss, macro_mAP))
        if macro_mAP > best_macro_mAP:
            best_macro_mAP = macro_mAP
            torch.save(model, log_name + "/epoch-best.pkl")
            best_epoch = epoch
        if epoch % args.save_interval == 0:
            print('saving the %d epoch' % (epoch))
            torch.save(model, log_name + "/epoch-%d.pkl" % (epoch))

    f_logger.write("best epoch num: %d" % best_epoch)
    f_logger.close()

    results = vars(args)
    results.update({'best_epoch_mAP': best_macro_mAP, 'best_epoch': best_epoch})

    with open(os.path.join(log_name, "train_info.json"), 'w') as f:
        json.dump(results, f, indent=2)

    if args.include_test:
        # from test_clip import test_main
        from test import test_main

        class test_ap(object):
            def __init__(self, args):
                self.model_path = log_name
                self.save_results = True
                self.dataloader_name = "trailer_multimodal_features"


        test_args = test_ap(args)
        test_main(test_args)

if __name__=="__main__":
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='trailer_visual', help='model name')
    parser.add_argument('--data_dir', type=str, default='data/CondensedMovies/MovieCLIP_features',
                        help='path to save data files')
    parser.add_argument('--visual_feature_dir', type=str,
                        default='data/CondensedMovies/MovieCLIP_features/Visual_features',
                        help='path to visual features')
    parser.add_argument('--visual_feature_version', type=str, default="ViT-B-32")
    parser.add_argument('--audio_feature_file', type=str, default=None)
    parser.add_argument('--text_token_file',type=str,
                        default=None)
    parser.add_argument('--text_feature_file',type=str,
                        default=None)
    parser.add_argument('--num_epoch', type=int, default=10, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512, help='dimensionality of hidden feature')
    parser.add_argument('--num_categories', type=int, default=21, help='The number of categories')
    parser.add_argument('--concatenate', type=str, default='mean', help='way of concatenation')
    parser.add_argument('--dropout', type=str2bool, default=True, help='use dropout or not')

    parser.add_argument('--include_test', type=str2bool, default=False, help='do test or not')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--save_interval', type=int, default=5, help='the interval between saved epochs')
    parser.add_argument('--process_interval', type=int, default=10, help='the interval between process print')

    args = parser.parse_args()
    print(args)

    main(args)