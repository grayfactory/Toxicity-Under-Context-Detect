import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd


def tokenize_segmented_text(text, parent, tokenizer, max_len):
  "text from DataFrame : text, parent"
  
  tokens_a = tokenizer.tokenize(parent) # parent context first
  tokens_b = tokenizer.tokenize(text)
  _max_length = max_len - 3 # preserved for [CLS],[SEP] tokens

  # simple huristic
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= _max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

  tokens = [tokenizer.cls_token, *tokens_a, tokenizer.sep_token]
  segment_ids = [0]*len(tokens)
  if tokens_b:
    tokens += [*tokens_b, tokenizer.sep_token]
    segment_ids += [1]*(len(tokens_b)+1)

  # mask has 1 for real tokens and - for padding tokens
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  attention_mask = [1] * len(tokens)

  while len(input_ids) < max_len:
    input_ids.append(0)
    attention_mask.append(0)
    segment_ids.append(0)
  
  encoding = dict()

  encoding['input_ids'] = input_ids
  encoding['attention_mask'] = attention_mask
  encoding['segment_ids'] = segment_ids

  return encoding

class TOXICDataset(Dataset):

  def __init__(self, text, label, tokenizer, max_len):
    self.text = text
    self.label = label
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.text)

  
  def __getitem__(self, idx):
    text = str(self.text.iloc[idx,0]) # [text, parent]
    parent = str(self.text.iloc[idx,1])
    label = self.label.iloc[idx]

    # custom encoding
    encoding = tokenize_segmented_text(text, parent, 
                                       tokenizer=self.tokenizer, 
                                       max_len=self.max_len)
    # dict 형태로 return
    return {
        'text' : text,
        'input_ids' : torch.tensor(encoding['input_ids'], dtype=torch.long).view(-1), # 1차원 배열로 [1,256] -> [256]
        'attention_mask' : torch.tensor(encoding['attention_mask'], dtype=torch.float32).view(-1),
        'token_type_ids' : torch.tensor(encoding['segment_ids'], dtype=torch.long).view(-1),
        'label' : torch.tensor(label, dtype=torch.long)
    }

def f1_score(y_true, y_pred):
  
  epsilon = 1e-7 
  # print(y_true, y_pred, y_true.shape, y_pred.shape)
  tp = (y_true * y_pred).sum().to(torch.float32)
  tn = ((1-y_true) * (1-y_pred)).sum().to(torch.float32)
  fp = ((1-y_true) * y_pred).sum().to(torch.float32)
  fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

  precision = tp / (tp + fp + epsilon)
  recall = tp / (tp + fn + epsilon)

  f1_value = 2* (precision*recall) / (precision + recall + epsilon)

  return f1_value


class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1): # n: batch_size ex 128
    self.val = val
    self.sum += val * n # 128 개 * value를 부풀려서
    self.count += n # batch 돌 때마다 n수 update
    self.avg = self.sum / self.count # 결국 total f1을 전체 data 길이로 나눠서 평균

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  model = model.train()
  losses = []
  correct_predictions = 0

  f1_score_meter = AverageMeter()

  # iteration by batch size
  for data in tqdm(data_loader):
    # input data
    input_ids = data['input_ids'].to(device)
    attention_masks = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    # true label
    labels = data['label'].to(device)

    y_pred = model(input_ids = input_ids, 
                  attention_mask=attention_masks,
                  token_type_ids=token_type_ids)[0]

    # print(y_pred, labels)
    loss = loss_fn(y_pred, labels)


    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    _, pred = torch.max(y_pred, dim=1) # indices [0,1] 만 받음, 앞은 max tensor값
    correct_predictions += torch.sum(pred == labels)
    losses.append(loss.item())

    f1_value = f1_score(labels, pred)
    f1_score_meter.update(f1_value, len(labels))

  # mean loss, number of correct_prediction (each batch)
  return correct_predictions.double() / n_examples, np.mean(losses), f1_score_meter


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0

  f1_score_meter = AverageMeter()

  with torch.no_grad():
    for data in tqdm(data_loader):

      input_ids = data['input_ids'].to(device)
      attention_masks = data['attention_mask'].to(device)
      token_type_ids = data['token_type_ids'].to(device)
      labels = data['label'].to(device)

      y_pred = model(input_ids = input_ids, 
                    attention_mask=attention_masks,
                    token_type_ids=token_type_ids)[0]

      _, pred = torch.max(y_pred, dim=1)
      loss = loss_fn(y_pred, labels)
      correct_predictions += torch.sum(pred == labels)
      losses.append(loss.item())

      f1_value = f1_score(labels, pred)
      f1_score_meter.update(f1_value, len(labels))

  return correct_predictions.double() / n_examples, np.mean(losses), f1_score_meter


def model_score(model, data_loader, configs):
  model = model.eval()

  y_pred_batch = []
  y_true_batch = []
  y_prob_batch = []

  with torch.no_grad():
    for data in tqdm(data_loader):

      input_ids = data['input_ids'].to(configs.device)
      attention_masks = data['attention_mask'].to(configs.device)
      token_type_ids = data['token_type_ids'].to(configs.device)
      labels = data['label'].to(configs.device)

      y_pred = model(input_ids = input_ids, 
                attention_mask=attention_masks,
                token_type_ids=token_type_ids)[0]
      # raw logit to softmax
      prob = F.softmax(y_pred, dim=1)

      _, pred = torch.max(y_pred, dim=1)

      y_pred_batch.append(pred.cpu().tolist())
      y_true_batch.append(labels.cpu().tolist())
      y_prob_batch.append(prob[:,1].cpu().tolist())

  return y_pred_batch, y_true_batch, y_prob_batch

