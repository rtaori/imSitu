import time
from torch import optim
import torch
import json
from tqdm import tqdm

from imsitu import imSituVerbRoleLocalNounEncoder 
from imsitu import imSituTensorEvaluation 
from imsitu import imSituSituation 
from imsitu import imSituSimpleImageFolder
from baseline_crf import baseline_crf, format_dict


def predict_human_readable (dataset_loader, simple_dataset,  model, outdir, top_k):
  model.eval()  
  print("predicting...")
  mx = len(dataset_loader) 
  preds = {}
  for i, (input, index) in enumerate(tqdm(dataset_loader)):
      # print ("{}/{} batches".format(i+1,mx))
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      (scores,predictions)  = model.forward_max(input_var)
      #(s_sorted, idx) = torch.sort(scores, 1, True)
      human = encoder.to_situation(predictions)
      (b,p,d) = predictions.size()
      for _b in range(0,b):
        items = []
        offset = _b *p
        for _p in range(0, p):
          items.append(human[offset + _p])
          items[-1]["score"] = scores.data[_b][_p].item()
        items = sorted(items, key = lambda x: -x["score"])[:top_k]
        name = simple_dataset.images[index[_b][0]].split(".")[:-1]
        preds[name[0]] = items
        # name.append("predictions")
        # outfile = outdir + '/' + ".".join(name)
  json.dump(preds, open(outdir + '/predictions.json', "w"))


def compute_features(dataset_loader, simple_dataset,  model, outdir):
  model.eval()  
  print ("computing features...")
  mx = len(dataset_loader) 
  for i, (input, index) in enumerate(dataset_loader):
      print ("{}/{} batches\r".format(i+1,mx)) ,
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      features  = model.forward_features(input_var).cpu().data
      b = index.size()[0]
      for _b in range(0,b):
        name = simple_dataset.images[index[_b][0]].split(".")[:-1]
        name.append("features")
        outfile = outdir + ".".join(name)
        torch.save(features[_b], outfile)
  print ("\ndone.")

def eval_model(dataset_loader, encoding, model):
    model.eval()
    print ("evaluating model...")
    top1 = imSituTensorEvaluation(1, 3, encoding)
    top5 = imSituTensorEvaluation(5, 3, encoding)
 
    mx = len(dataset_loader) 
    for i, (index, input, target) in enumerate(tqdm(dataset_loader)):
      # print ("{}/{} batches\r".format(i+1,mx)) ,
      input_var = torch.autograd.Variable(input.cuda(), volatile = True)
      target_var = torch.autograd.Variable(target.cuda(), volatile = True)
      (scores,predictions)  = model.forward_max(input_var)
      (s_sorted, idx) = torch.sort(scores, 1, True)
      top1.add_point(target, predictions.data, idx.data)
      top5.add_point(target, predictions.data, idx.data)
      
    print ("\ndone.")
    return (top1, top5) 

def train_model(max_epoch, eval_frequency, train_loader, dev_loader, model, encoding, optimizer, save_dir, timing = False): 
    model.train()

    time_all = time.time()

    pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    top1 = imSituTensorEvaluation(1, 3, encoding)
    top5 = imSituTensorEvaluation(5, 3, encoding)
    loss_total = 0 
    print_freq = 10
    total_steps = 0
    avg_scores = []
  
    for k in range(0,max_epoch):  
      for i, (index, input, target) in enumerate(train_loader):
        total_steps += 1
   
        t0 = time.time()
        t1 = time.time() 
      
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())
        (_,v,vrn,norm,scores,predictions)  = pmodel(input_var)
        (s_sorted, idx) = torch.sort(scores, 1, True)
        #print norm 
        if timing : print ("forward time = {}".format(time.time() - t1))
        optimizer.zero_grad()
        t1 = time.time()
        loss = model.mil_loss(v,vrn,norm, target, 3)
        if timing: print ("loss time = {}".format(time.time() - t1))
        t1 = time.time()
        loss.backward()
        #print loss
        if timing: print ("backward time = {}".format(time.time() - t1))
        optimizer.step()
        loss_total += loss.item()
        #score situation
        t2 = time.time() 
        top1.add_point(target, predictions.data, idx.data)
        top5.add_point(target, predictions.data, idx.data)
     
        if timing: print ("eval time = {}".format(time.time() - t2))
        if timing: print ("batch time = {}".format(time.time() - t0))
        if total_steps % print_freq == 0:
           top1_a = top1.get_average_results()
           top5_a = top5.get_average_results()
           print ("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}, batch time = {:.2f}".format(total_steps-1,k,i, format_dict(top1_a, "{:.2f}", "1-"), format_dict(top5_a,"{:.2f}","5-"), loss.item(), loss_total / ((total_steps-1)%eval_frequency) , (time.time() - time_all)/ ((total_steps-1)%eval_frequency)))
        if total_steps % eval_frequency == 0:
          print ("eval..."    )
          etime = time.time()
          (top1, top5) = eval_model(dev_loader, encoding, model)
          model.train() 
          print ("... done after {:.2f} s".format(time.time() - etime))
          top1_a = top1.get_average_results()
          top5_a = top5.get_average_results()

          avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + top5_a["value"] + top5_a["value-all"] + top5_a["value*"] + top5_a["value-all*"]
          avg_score /= 8

          print ("Dev {} average :{:.2f} {} {}".format(total_steps-1, avg_score*100, format_dict(top1_a,"{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-")))
          
          avg_scores.append(avg_score)
          maxv = max(avg_scores)

          if maxv == avg_scores[-1]: 
            torch.save(model.state_dict(), save_dir + "/{0}.model".format(maxv))   
            print ("new best model saved! {0}".format(maxv))

          top1 = imSituTensorEvaluation(1, 3, encoding)
          top5 = imSituTensorEvaluation(5, 3, encoding)
          loss_total = 0
          time_all = time.time()

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="imsitu Situation CRF. Training, evaluation, prediction and features.") 
  parser.add_argument("--output_dir", help="location to put output, such as models, features, predictions")
  parser.add_argument("--image_dir", default="./resized_256", help="location of images to process")
  parser.add_argument("--dataset_dir", default="./", help="location of train.json, dev.json, ect.") 
  parser.add_argument("--weights_file", help="the model to start from")
  parser.add_argument("--encoding_file", help="a file corresponding to the encoder")
  parser.add_argument("--cnn_type", choices=["resnet_34", "resnet_50", "resnet_101"], default="resnet_101", help="the cnn to initilize the crf with") 
  parser.add_argument("--batch_size", default=64, help="batch size for training", type=int)
  parser.add_argument("--learning_rate", default=1e-5, help="learning rate for ADAM", type=float)
  parser.add_argument("--weight_decay", default=5e-4, help="learning rate decay for ADAM", type=float)  
  parser.add_argument("--eval_frequency", default=500, help="evaluate on dev set every N training steps", type=int) 
  parser.add_argument("--training_epochs", default=20, help="total number of training epochs", type=int)
  parser.add_argument("--eval_file", default="dev.json", help="the dataset file to evaluate on, ex. dev.json test.json")
  parser.add_argument("--top_k", default="10", type=int, help="topk to use for writing predictions to file")

  args = parser.parse_args()

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
  
  dataset_train = imSituSituation(args.image_dir, train_set, encoder, model.train_preprocess())
  dataset_dev = imSituSituation(args.image_dir, dev_set, encoder, model.dev_preprocess())

  device_array = [i for i in range(0,ngpus)]
  batch_size = args.batch_size*ngpus

  train_loader  = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4) 
  dev_loader  = torch.utils.data.DataLoader(dataset_dev, batch_size = batch_size, shuffle = True, num_workers = 4) 

  model.cuda()
  optimizer = optim.Adam(model.parameters(), lr = args.learning_rate , weight_decay = args.weight_decay)
  train_model(args.training_epochs, args.eval_frequency, train_loader, dev_loader, model, encoder, optimizer, args.output_dir)
  

  # elif args.command == "eval":
  #   print ("command = evaluating")
  #   eval_file = json.load(open(args.dataset_dir + "/" + args.eval_file))  
      
  #   if args.encoding_file is None: 
  #     print ("expecting encoder file to run evaluation")
  #     exit()
  #   else:
  #     encoder = torch.load(args.encoding_file)
  #   print ("creating model...")
  #   model = baseline_crf(encoder, cnn_type = args.cnn_type)
    
  #   if args.weights_file is None:
  #     print ("expecting weight file to run features")
  #     exit()
    
  #   print ("loading model weights...")
  #   model.load_state_dict(torch.load(args.weights_file))
  #   model.cuda()
    
  #   dataset = imSituSituation(args.image_dir, eval_file, encoder, model.dev_preprocess())
  #   loader  = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4) 

  #   with torch.no_grad():
  #     (top1, top5) = eval_model(loader, encoder, model)    
  #   top1_a = top1.get_average_results()
  #   top5_a = top5.get_average_results()

  #   avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + top5_a["value"] + top5_a["value-all"] + top5_a["value*"] + top5_a["value-all*"]
  #   avg_score /= 8

  #   print ("Average :{:.2f} {} {}".format(avg_score*100, format_dict(top1_a,"{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-")))
       

  # elif args.command == "predict":
  #   print ("command = predict")
  #   if args.encoding_file is None: 
  #     print ("expecting encoder file to run features")
  #     exit()
  #   else:
  #     encoder = torch.load(args.encoding_file)
 
  #   print ("creating model..." )
  #   model = baseline_crf(encoder, cnn_type = args.cnn_type)
 
  #   if args.weights_file is None:
  #     print ("expecting weight file to run features")
  #     exit()
    
  #   print ("loading model weights...")
  #   model.load_state_dict(torch.load(args.weights_file))
  #   model.cuda()

  #   eval_file = json.load(open(args.dataset_dir + "/" + args.eval_file))
  #   eval_images = set(eval_file.keys())
  #   selection_func = lambda x: x.split('/')[-1] in eval_images

  #   folder_dataset = imSituSimpleImageFolder(args.image_dir, model.dev_preprocess(), selection_func=selection_func)
  #   image_loader  = torch.utils.data.DataLoader(folder_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 4) 
    
  #   with torch.no_grad():
  #     predict_human_readable(image_loader, folder_dataset, model, args.output_dir, args.top_k)
