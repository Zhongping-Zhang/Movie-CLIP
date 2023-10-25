import torch
import torch.nn.functional as F

import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from dataloader.dataset import (trailer_multimodal_features,)
from sklearn.metrics import precision_score, recall_score, average_precision_score

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


dataloader_dict = {
    "trailer_multimodal_features": trailer_multimodal_features,
}


def test(args, config):
    dataloader = dataloader_dict[args.dataloader_name]

    if config['visual_feature_dir'] is not None and config['audio_feature_file'] is None and config['text_token_file'] is None: # visual only
        testdata = dataloader(phase='test',
                              data_dir=config['data_dir'],
                              visual_feature_dir=config['visual_feature_dir'],
                              visual_feature_file=config['visual_feature_version']+"_test.pkl",
                              num_classes=config["num_categories"],
                              )

    elif config['audio_feature_file'] is not None and config['text_token_file'] is None: # audio/audio+visual
        testdata = dataloader(phase='test',
                              data_dir=config['data_dir'],
                              visual_feature_dir=config['visual_feature_dir'],
                              visual_feature_file=config['visual_feature_version']+"_test.pkl" if config['visual_feature_version'] else None,
                              audio_feature_file=config['audio_feature_file'],
                              num_classes=config["num_categories"],
                              )
    elif config['text_token_file'] is not None: # text, text+visual, text+visual+audio
        testdata = dataloader(phase='test',
                              data_dir=config['data_dir'],
                              visual_feature_dir=config['visual_feature_dir'],
                              visual_feature_file=config['visual_feature_version']+"_test.pkl" if config['visual_feature_version'] else None,
                              audio_feature_file=config['audio_feature_file'],
                              text_token_file=config['text_token_file'],
                              text_feature_file=config['text_feature_file'],
                              num_classes=config["num_categories"],
                              )
    else:
        assert False, "invalid dataloader"

    test_loader = torch.utils.data.DataLoader(testdata, batch_size=config['batch_size'], shuffle=False,
                                              num_workers=config['n_cpu'])
    model = torch.load(os.path.join(args.model_path, "epoch-best.pkl"))
    return_predictions = []
    return_labels = []

    with torch.no_grad():
        model.eval()  # model.eavl() fix the BN and Dropout
        test_loss = 0
        for batch_idx, sample in enumerate(test_loader):
            movie_id, label = sample['movie_id'],sample['label_onehot']
            label = label.type(LongTensor)  # convert to gpu computation

            if config['visual_feature_dir'] is not None and config["audio_feature_file"] is None and config["text_token_file"] is None: # visual-only
                visual_feature = sample['visual_feature'].to(device).type(FloatTensor)
                output = model(visual_feature)["multimodal_output"]
            elif config["audio_feature_file"] is not None and config["visual_feature_dir"] is None and config["text_token_file"] is None: # audio-only
                audio_feature = sample['audio_feature'].to(device).type(FloatTensor)
                output = model(audio_feature)["multimodal_output"]
            elif config["text_token_file"] is not None and config["visual_feature_dir"] is None and config["audio_feature_file"] is None: # text-only
                text_feature = sample['text_feature'].to(device).type(FloatTensor)
                output = model(text_feature)["multimodal_output"]
            elif config['visual_feature_dir'] is not None and config["audio_feature_file"] is not None and config["text_token_file"] is None: # visual+audio
                visual_feature = sample['visual_feature'].to(device).type(FloatTensor)
                audio_feature = sample['audio_feature'].to(device).type(FloatTensor)
                output = model(visual_feature,audio_feature)["multimodal_output"]
            elif config['visual_feature_dir'] is not None and config['audio_feature_file'] is not None and config["text_token_file"] is not None: # visual+audio+text
                visual_feature = sample['visual_feature'].to(device).type(FloatTensor)
                audio_feature = sample['audio_feature'].to(device).type(FloatTensor)
                text_feature = sample['text_feature'].to(device).type(FloatTensor)
                output = model(visual_feature,audio_feature,text_feature)["multimodal_output"]

            test_loss += F.binary_cross_entropy_with_logits(output, label.float())
            predicted = output.data
            predicted_np = predicted.cpu().numpy()
            label_np = label.cpu().numpy()

            return_predictions.append(predicted_np)
            return_labels.append(label_np)

    return np.concatenate(return_predictions), np.concatenate(return_labels)


def test_main(args):
    config_file = os.path.join(args.model_path, "train_info.json")
    with open(config_file, 'r') as f:
        config = json.load(f)

    outputs, labels = test(args, config)
    outputs_sigmoid = sigmoid(outputs)
    print("outputs shape:", outputs.shape, labels.shape)
    average_choice = ['micro', 'samples', 'weighted', 'macro']
    mAP = {}
    precision = {}
    recall = {}
    for i in range(4):
        mAP[average_choice[i]] = average_precision_score(labels[:, :], outputs_sigmoid[:, :], average=average_choice[i])
        precision[average_choice[i]] = precision_score(labels, outputs_sigmoid >= 0.5, average=average_choice[i])
        recall[average_choice[i]] = recall_score(labels, outputs_sigmoid >= 0.5, average=average_choice[i])

    print("recall-macro: ", recall['macro'])
    print("precision-macro: ", precision['macro'])
    print("mAP-macro: ", mAP['macro'])

    print("recall-micro: ", recall['micro'])
    print("precision-micro: ", precision['micro'])
    print("mAP-micro: ", mAP['micro'])

    # curve_precision, curve_recall, curve_thresholds = precision_recall_curve(labels, outputs_sigmoid)

    """ calculate precision and recall for each category outputs@0.5"""
    outputs_prediction = outputs_sigmoid >= 0.5

    precisions = {}
    recalls = {}
    for j in range(outputs.shape[1]):
        ptotal = 0
        pcorrect = 0
        rtotal = 0
        rcorrect = 0
        for i in range(outputs.shape[0]):
            if outputs_prediction[i][j]:
                ptotal += 1
                if labels[i][j]:
                    pcorrect += 1

            if labels[i][j]:
                rtotal += 1
                if outputs_prediction[i][j]:
                    rcorrect += 1

        # assert rtotal!=0
        # assert ptotal!=0
        # print(pcorrect/ptotal)
        precisions[j] = -0.01 if ptotal == 0 else pcorrect / ptotal
        recalls[j] = -0.01 if rtotal == 0 else rcorrect / rtotal

    with open(os.path.join(config['data_dir'], "vocab/vocab_genres21.json"), "r") as f:
        vocab_genre21 = json.load(f)

    plt.figure()
    plt.bar(vocab_genre21.keys(), precisions.values())
    plt.xticks(rotation=65)
    plt.title('precisions')
    if args.save_results:
        plt.savefig(os.path.join(args.model_path, "precision_percategory.png"), bbox_inches='tight')

    plt.figure()
    plt.bar(vocab_genre21.keys(), recalls.values())
    plt.xticks(rotation=65)
    plt.title('recalls')
    if args.save_results:
        plt.savefig(os.path.join(args.model_path, "recall_percategory.png"), bbox_inches='tight')

    if args.save_results:
        test_results = {}
        test_results['best epoch'] = config['best_epoch']
        test_results['mAP'] = mAP
        test_results['precision'] = precision
        test_results['recall'] = recall
        test_results['precision per genre'] = precisions
        test_results['recall per genre'] = recalls
        with open(os.path.join(args.model_path, 'testing_results.json'), 'w') as ft:
            json.dump(test_results, ft, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='logs/trailer_clip', help='path to trained model')
    parser.add_argument('--dataloader_name', type=str, default="trailer_clipfeat", help="specify a dataloader")
    parser.add_argument('--save_results', type=bool, default=True, help='dump test results in a json file')
    args = parser.parse_args()
    print(args)
    test_main(args)