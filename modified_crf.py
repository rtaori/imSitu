import os
import time
from torch import optim
import torch
import json
import argparse
from tqdm import tqdm
import subprocess

from imsitu import imSituTensorEvaluation
from imsitu import imSituSituation
from baseline_crf import baseline_crf, format_dict
from utils import collapse_annotations


def eval_model(dataset_loader, encoding, model):
    model.eval()
    print("evaluating model...")
    top1 = imSituTensorEvaluation(1, 3, encoding)

    for _, input, target in tqdm(dataset_loader):
        input_var = torch.autograd.Variable(input.cuda())
        (scores, predictions) = model.forward_max(input_var)
        (s_sorted, idx) = torch.sort(scores, 1, True)
        top1.add_point(target, predictions.data, idx.data)

    return top1


def predict_human_readable(dataset_loader, encoding, model, top_k):
    model.eval()
    print("predicting...")
    preds = {}

    for ids, input, target in tqdm(dataset_loader):
        input_var = torch.autograd.Variable(input.cuda())
        (scores, predictions) = model.forward_max(input_var)
        human = encoding.to_situation(predictions)
        (b, p, d) = predictions.size()
        
        for _b in range(0, b):
            items = []
            offset = _b * p
            for _p in range(0, p):
                items.append(human[offset + _p])
                items[-1]["score"] = scores.data[_b][_p].item()
            items = sorted(items, key=lambda x: -x["score"])[:top_k]
            name = ids[_b].split(".")[:-1]
            preds[name[0]] = items

    return preds


def train_model(max_epoch, train_loader, dev_loader, model, encoding, optimizer, save_path, n_refs=3):
    time_all = time.time()
    print_freq = 100

    for k in range(0, max_epoch):
        top1 = imSituTensorEvaluation(1, n_refs, encoding)
        loss_total = 0
        model.train()

        for i, (_, input, target) in enumerate(tqdm(train_loader)):
            input_var = torch.autograd.Variable(input.cuda())
            (_, v, vrn, norm, scores, predictions) = model(input_var)
            (s_sorted, idx) = torch.sort(scores, 1, True)

            optimizer.zero_grad()
            loss = model.mil_loss(v, vrn, norm, target, n_refs)
            loss.backward()

            optimizer.step()
            loss_total += loss.item()

            top1.add_point(target, predictions.data, idx.data)

            if i % print_freq == 0:
                top1_a = top1.get_average_results()
                print(f"Epoch {k} [{i}/{len(train_loader)}], {format_dict(top1_a, '{:.2f}', '1-')}, "
                      f"loss = {loss.item():.2f}, avg_loss = {loss_total/(i+1):.2f}, "
                      f"batch_time = {(time.time()-time_all)/(i+1):.2f}")

        with torch.no_grad():
            top1 = eval_model(dev_loader, encoding, model)
        top1_a = top1.get_average_results()
        avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top1_a["value*"] + top1_a["value-all*"]
        avg_score /= 5
        print(f"Epoch {k} Dev: average = {avg_score*100:.2f}, {format_dict(top1_a, '{:.2f}', '1-')}")

        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imsitu Situation CRF. Training, evaluation, prediction and features.")
    parser.add_argument("--save_model_path", default="model.pth", help="location to put model")
    parser.add_argument("--save_preds_path", default="predictions.json", help="location to put model predictions")
    parser.add_argument("--image_dir", default="/scr/biggest/of500_images_resized",
                        help="location of images to process")
    parser.add_argument("--encoding_file", default="/juice/scr/rtaori/imsitu_data/baseline_models/baseline_encoder",
                        help="a file corresponding to the encoder")
    parser.add_argument("--cnn_type", choices=["resnet_34", "resnet_50", "resnet_101"],
                        default="resnet_34", help="the cnn to initilize the crf with")
    parser.add_argument("--batch_size", default=64,
                        help="batch size for training", type=int)
    parser.add_argument("--learning_rate", default=1e-5,
                        help="learning rate for ADAM", type=float)
    parser.add_argument("--weight_decay", default=5e-4,
                        help="learning rate decay for ADAM", type=float)
    parser.add_argument("--training_epochs", default=30,
                        help="total number of training epochs", type=int)
    parser.add_argument("--top_k", default=1, type=int,
                        help="topk to use for writing predictions to file")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="number of dataloading workers")
    parser.add_argument("--collapse_annotations", choices=["majority", "random"],
                        help="keep full annotations or collapse into one by majority or randomly")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        subprocess.run('cp /u/scr/nlp/data/of500_images_resized.tar /scr/biggest'.split(' '))
        subprocess.run('tar -xf /scr/biggest/of500_images_resized.tar -C /scr/biggest'.split(' '))

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = True

    encoder = torch.load(args.encoding_file)
    model = baseline_crf(encoder, cnn_type=args.cnn_type, ngpus=1)
    model = model.cuda()

    train_set = json.load(open("train.json"))
    dev_set = json.load(open("dev.json"))
    test_set = json.load(open("test.json"))
    dataset = train_set | dev_set | test_set

    if args.collapse_annotations == "majority":
        train_set = collapse_annotations(train_set, use_majority=True)
    elif args.collapse_annotations == "random":
        train_set = collapse_annotations(train_set, use_majority=False)
    dataset_train = imSituSituation(args.image_dir, train_set, encoder, model.train_preprocess())
    dataset_dev = imSituSituation(args.image_dir, dev_set, encoder, model.dev_preprocess())

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)
    dev_loader = torch.utils.data.DataLoader(dataset_dev, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_model(args.training_epochs, train_loader, dev_loader, model, encoder, optimizer,
                args.save_model_path, n_refs=1 if args.collapse_annotations else 3)

    with torch.no_grad():
        preds = predict_human_readable(dev_loader, encoder, model, args.top_k)
    json.dump(preds, open(args.save_preds_path, "w"))
