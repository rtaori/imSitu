import time
import json
import argparse
from tqdm import tqdm
import numpy as np
import wandb
import torch
from torch import optim

from imsitu import imSituTensorEvaluation
from imsitu import imSituSituation
from baseline_crf import baseline_crf, format_dict
import data


def predict_human_readable(dataset_loader, encoding, model, top_k):
    model.eval()
    preds = {}
    top1 = imSituTensorEvaluation(1, 3, encoding)

    for ids, input, target in tqdm(dataset_loader):
        input_var = torch.autograd.Variable(input.cuda())
        (scores, predictions) = model.forward_max(input_var)
        (s_sorted, idx) = torch.sort(scores, 1, True)
        top1.add_point(target, predictions.data, idx.data)

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

    return top1, preds


def train_model(max_epoch, train_loader, model, encoding, optimizer, n_refs=3):
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
            # wandb.log({"loss": loss.item()})

            if i % print_freq == 0 and i > 0:
                top1_a = top1.get_average_results()
                print(f"Epoch {k} [{i}/{len(train_loader)}], {format_dict(top1_a, '{:.2f}', '1-')}, "
                      f"loss = {loss.item():.2f}, avg_loss = {loss_total/(i+1):.2f}, "
                      f"batch_time = {(time.time()-time_all)/(i+1):.2f}")


def calculate_training_epochs(dataset_len):
    if dataset_len <= 20000:
        return 50
    elif 20000 < dataset_len <= 35000:
        return int(50 - 10 * (dataset_len - 20000) / 15000)
    elif 35000 < dataset_len <= 50000:
        return int(40 - 5 * (dataset_len - 35000) / 15000)
    elif 50000 < dataset_len <= 75000:
        return int(35 - 5 * (dataset_len - 50000) / 25000)
    elif 75000 < dataset_len:
        return 30


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imsitu Situation CRF. Training, evaluation, prediction and features.")
    parser.add_argument('--wandb-group')
    parser.add_argument("--save-dir", default="/juice/scr/rtaori/imsitu_feedback")
    parser.add_argument("--image-dir", default="/scr/biggest/of500_images_resized")
    parser.add_argument("--encoding-file", default="baseline_encoder")
    parser.add_argument("--cnn-type", choices=["resnet_18", "resnet_34", "resnet_50", "resnet_101"], default="resnet_18")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--collapse-annotations", choices=["majority", "random"], default='majority',
                        help="keep full annotations or collapse into one by majority or randomly")
    parser.add_argument("--init-train-set-size", default=20000, type=int)
    parser.add_argument('--num-rounds', type=int, default=25)
    parser.add_argument("--test-set-imgs-per-class", default=50, type=int)
    parser.add_argument('--new-label-samples', type=int, default=5000)
    parser.add_argument('--new-unlabel-samples', type=int, default=5000)
    parser.add_argument('--filter-out-low-scores', action='store_true')
    parser.add_argument('--filter-score-threshold', type=float, default=1)
    parser.add_argument('--model-prediction-type', choices=["max_max", "max_marginal"], default="max_max")

    args = parser.parse_args()

    mode = 'disabled' if not args.wandb_group else 'online'
    wandb.init(project='feedback-imsitu', entity='hashimoto-group', group=args.wandb_group, mode=mode)
    wandb.config.update(vars(args))

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = True

    data.check_download_data(args.image_dir)

    train_set = json.load(open("train.json"))
    dev_set = json.load(open("dev.json"))
    test_set = json.load(open("test.json"))
    dataset = train_set | dev_set | test_set

    test_set, reserve_set = data.rand_split_test_set(dataset, args.test_set_imgs_per_class)
    if args.collapse_annotations in ["majority", "random"]:
        reserve_set = data.collapse_annotations(reserve_set, use_majority=args.collapse_annotations == "majority")
    train_set, reserve_set = data.rand_split_dataset(reserve_set, args.init_train_set_size)

    encoder = torch.load(args.encoding_file)

    for round in range(args.num_rounds):
        # initialize model and optimizer
        model = baseline_crf(encoder, cnn_type=args.cnn_type, ngpus=1, prediction_type=args.model_prediction_type)
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        num_epochs = calculate_training_epochs(len(train_set))

        # create dataloaders
        dataset_train = imSituSituation(args.image_dir, train_set, encoder, model.train_preprocess())
        dataset_test = imSituSituation(args.image_dir, test_set, encoder, model.dev_preprocess())
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, 
                                                   shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.num_workers)

        print(f'*** ROUND {round} ***')
        stats = {'round': round, 'train_epochs': num_epochs,
                 'size_train_set': len(train_set), 'size_reserve_set': len(reserve_set), 'size_test_set': len(test_set)}
        print(stats)

        train_model(num_epochs, train_loader, model, encoder, optimizer, n_refs=1 if args.collapse_annotations else 3)

        # evaluate model on test set and get results
        with torch.no_grad():
            top1, preds = predict_human_readable(test_loader, encoder, model, top_k=1)
        top1_a = top1.get_average_results()
        top1_a['avg-score'] = np.mean([v for v in top1_a.values()])

        # log results, including a histogram of predicted scores
        stats = stats | {'eval/'+k:v for k, v in top1_a.items()}        
        scores_table = wandb.Table(data=[[v[0]['score']] for v in preds.values()], columns=['scores'])
        scores_histogram = wandb.plot.histogram(scores_table, 'scores', title='test set scores histogram')
        stats = stats | {'scores_histogram': scores_histogram}

        # log gender bias metadata about the datasets and predictions
        preds_trans = data.transform_preds_to_dataset(preds)
        preds_stats = data.get_dataset_gender_stats(preds_trans)
        train_stats = data.get_dataset_gender_stats(train_set)
        test_stats = data.get_dataset_gender_stats(test_set)
        stats = stats | {'preds/'+k:v for k, v in preds_stats.items()}
        stats = stats | {'train/'+k:v for k, v in train_stats.items()}
        stats = stats | {'test/'+k:v for k, v in test_stats.items()}

        if len(reserve_set) < args.new_label_samples + args.new_unlabel_samples:
            print(f'Ending retraining - not enough remaining samples in reserve set ({len(reserve_set)} left).')
            wandb.log(stats)
            torch.save(model.state_dict(), f'{args.save_dir}/models/{wandb.run.name}_final.pth')

        if args.new_label_samples > 0:
            reserve_set_selected, reserve_set = data.rand_split_dataset(reserve_set, args.new_label_samples)
            train_set = train_set | reserve_set_selected

        if args.new_unlabel_samples > 0:
            reserve_set_partition, reserve_set = data.rand_split_dataset(reserve_set, args.new_unlabel_samples)
            dataset_reserve = imSituSituation(args.image_dir, reserve_set_partition, encoder, model.dev_preprocess())
            reserve_loader = torch.utils.data.DataLoader(dataset_reserve, batch_size=args.batch_size,
                                                         shuffle=False, num_workers=args.num_workers)

            # predict on unlabeled set, optionally filter by score
            with torch.no_grad():
                _, reserve_preds = predict_human_readable(reserve_loader, encoder, model, top_k=1)
            if args.filter_out_low_scores:
                reserve_preds = {k:v for k, v in reserve_preds.items() if v[0]['score'] >= args.filter_score_threshold}
                stats = stats | {'filter/frac_examples_accepted': len(reserve_preds) / args.new_unlabel_samples}

            # pseudo-labeling and add to train set for next round
            reserve_preds_trans = data.transform_preds_to_dataset(reserve_preds)
            assert all(k in reserve_set_partition for k in reserve_preds_trans.keys())
            train_set = train_set | reserve_preds_trans

        # save logs and model and dataset checkpoint
        wandb.log(stats)
        torch.save({'round': round+1, 'model_state_dict': model.state_dict(),
                    'train_set': train_set, 'reserve_set': reserve_set, 'test_set': test_set}, 
                   f'{args.save_dir}/models/{wandb.run.name}.pth')
