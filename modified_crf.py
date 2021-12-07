import time
from torch import optim
import torch
import json
import argparse
from tqdm import tqdm

from imsitu import imSituVerbRoleLocalNounEncoder 
from imsitu import imSituTensorEvaluation 
from imsitu import imSituSituation 
from baseline_crf import baseline_crf, format_dict
from utils import collapse_annotations


def eval_model(dataset_loader, encoding, model):
    model.eval()
    print ("evaluating model...")
    top1 = imSituTensorEvaluation(1, 3, encoding)
    top5 = imSituTensorEvaluation(5, 3, encoding)
 
    for index, input, target in tqdm(dataset_loader):
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      (scores,predictions)  = model.forward_max(input_var)
      (s_sorted, idx) = torch.sort(scores, 1, True)
      top1.add_point(target, predictions.data, idx.data)
      top5.add_point(target, predictions.data, idx.data)
      
    return (top1, top5) 


def train_model(max_epoch, eval_frequency, train_loader, dev_loader, model, encoding, optimizer, save_path, n_refs=3): 
    time_all = time.time()
    pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    print_freq = 100
    total_steps = 0
  
    for k in range(0,max_epoch):  
      top1 = imSituTensorEvaluation(1, n_refs, encoding)
      top5 = imSituTensorEvaluation(5, n_refs, encoding)
      loss_total = 0
      model.train()

      for i, (index, input, target) in enumerate(tqdm(train_loader)):
        total_steps += 1
         
        input_var = torch.autograd.Variable(input.cuda())
        (_,v,vrn,norm,scores,predictions)  = pmodel(input_var)
        (s_sorted, idx) = torch.sort(scores, 1, True)

        optimizer.zero_grad()
        loss = model.mil_loss(v,vrn,norm, target, n_refs)
        loss.backward()

        optimizer.step()
        loss_total += loss.item()

        top1.add_point(target, predictions.data, idx.data)
        top5.add_point(target, predictions.data, idx.data)
     
        if total_steps % print_freq == 0:
          top1_a = top1.get_average_results()
          top5_a = top5.get_average_results()
          print ("Epoch {} [{}/{}], {} , {}, loss = {:.2f}, avg loss = {:.2f}, batch time = {:.2f}".format(
            k, i, len(train_loader),
            format_dict(top1_a, "{:.2f}", "1-"), 
            format_dict(top5_a,"{:.2f}","5-"), 
            loss.item(), 
            loss_total / ((total_steps-1)%eval_frequency) , 
            (time.time() - time_all)/ ((total_steps-1)%eval_frequency))
          )
        
        if i == 10: break
              
      top1, top5 = eval_model(dev_loader, encoding, model)
      top1_a = top1.get_average_results()
      top5_a = top5.get_average_results()
      avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + \
                  top5_a["verb"] + top5_a["value"] + top5_a["value-all"] + top5_a["value*"] + top5_a["value-all*"]
      avg_score /= 8
      print("Dev {} average :{:.2f} {} {}".format(total_steps-1, avg_score*100, format_dict(top1_a,"{:.2f}", "1-"), 
            format_dict(top5_a, "{:.2f}", "5-")))

      torch.save(model.state_dict(), save_path+f'_ep{k}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="imsitu Situation CRF. Training, evaluation, prediction and features.") 
  parser.add_argument("--save_path", help="location to put model")
  parser.add_argument("--image_dir", default="./resized_256", help="location of images to process")
  parser.add_argument("--dataset_dir", default="./", help="location of train.json, dev.json, ect.") 
  parser.add_argument("--weights_file", help="the model to start from")
  parser.add_argument("--encoding_file", help="a file corresponding to the encoder")
  parser.add_argument("--cnn_type", choices=["resnet_34", "resnet_50", "resnet_101"], default="resnet_101", help="the cnn to initilize the crf with") 
  parser.add_argument("--batch_size", default=64, help="batch size for training", type=int)
  parser.add_argument("--learning_rate", default=1e-5, help="learning rate for ADAM", type=float)
  parser.add_argument("--weight_decay", default=5e-4, help="learning rate decay for ADAM", type=float)  
  parser.add_argument("--eval_frequency", default=1182, help="evaluate on dev set every N training steps", type=int) 
  parser.add_argument("--training_epochs", default=20, help="total number of training epochs", type=int)
  parser.add_argument("--eval_file", default="dev.json", help="the dataset file to evaluate on, ex. dev.json test.json")
  parser.add_argument("--top_k", default="10", type=int, help="topk to use for writing predictions to file")
  parser.add_argument("--num_workers", default=4, type=int, help="number of dataloading workers")
  parser.add_argument("--collapse_annotations", action="store_true", help="collapse train annotations into one")
  

  args = parser.parse_args()

  torch.multiprocessing.set_sharing_strategy('file_system')
  torch.backends.cudnn.benchmark = True

  train_set = json.load(open(args.dataset_dir+"/train.json"))
  dev_set = json.load(open(args.dataset_dir+"/dev.json"))

  if args.encoding_file is None: 
    encoder = imSituVerbRoleLocalNounEncoder(train_set)
    torch.save(encoder, args.output_dir + "/encoder")
  else:
    encoder = torch.load(args.encoding_file)

  ngpus = 1
  model = baseline_crf(encoder, cnn_type = args.cnn_type, ngpus = ngpus)
  
  if args.weights_file is not None:
    model.load_state_dict(torch.load(args.weights_file))
  
  if args.collapse_annotations:
    train_set = collapse_annotations(train_set)
  dataset_train = imSituSituation(args.image_dir, train_set, encoder, model.train_preprocess())
  dataset_dev = imSituSituation(args.image_dir, dev_set, encoder, model.dev_preprocess())

  device_array = [i for i in range(0,ngpus)]
  batch_size = args.batch_size*ngpus

  train_loader  = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = args.num_workers) 
  dev_loader  = torch.utils.data.DataLoader(dataset_dev, batch_size = batch_size, shuffle = True, num_workers = args.num_workers) 

  model.cuda()
  optimizer = optim.Adam(model.parameters(), lr = args.learning_rate , weight_decay = args.weight_decay)
  train_model(args.training_epochs, args.eval_frequency, train_loader, dev_loader, model, encoder, optimizer, 
              args.save_path, n_refs = 1 if args.collapse_annotations else 3)
