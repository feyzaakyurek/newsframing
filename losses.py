import torch
import torch.nn as nn

def FocalLoss(logits, labels, inverse_normed_freqs):
    labels = labels.type(torch.float32)
    probs = torch.sigmoid(logits)
    pt = (1 - labels) * (1 - probs) + labels * probs
    log_pt = torch.log(pt)
    floss = - (1 - pt)**2 * log_pt
    floss_weighted =  floss * inverse_normed_freqs 
    return torch.mean(floss_weighted)


def SoftmaxFocalLoss(logits, labels, inverse_normed_freqs):
    labels = labels.type(torch.float32)
    m = nn.Softmax(dim=1)
    probs = m(logits)
    logprobs = torch.log(probs) 
    logprobs = (1 - probs)**2 * logprobs 
    logyhat_for_gold = labels * logprobs * inverse_normed_freqs
    logyhat_for_gold_summed = torch.sum(logyhat_for_gold, dim=1)
    return torch.mean(-logyhat_for_gold_summed)


def SoftmaxLoss(logits, labels, inverse_normed_freqs):
    labels = labels.type(torch.float32)
    m = nn.Softmax(dim=1)
    probs = m(logits)
    logyhat_for_gold = labels * torch.log(probs)
    logyhat_for_gold_summed = torch.sum(logyhat_for_gold, dim=1)
    return torch.mean(-logyhat_for_gold_summed)

def SoftmaxWeightedLoss(logits, labels, inverse_normed_freqs):
    labels = labels.type(torch.float32)
    m = nn.Softmax(dim=1)
    probs = m(logits)
    logyhat_for_gold = labels * torch.log(probs) * inverse_normed_freqs
    logyhat_for_gold_summed = torch.sum(logyhat_for_gold, dim=1)
    return torch.mean(-logyhat_for_gold_summed)

def NormalizedLogSoftmaxLoss(logits, labels, inverse_normed_freqs):
    labels = labels.type(torch.float32)
    m = nn.Softmax(dim=1)
    probs = m(logits)
    logyhat_for_gold = labels * torch.log(probs)
    logyhat_for_gold_normalized_summed = torch.sum(logyhat_for_gold / labels.sum(dim=1).reshape((-1,1)), dim=1)
    return torch.mean(-logyhat_for_gold_normalized_summed)

def LogNormalizedSoftmaxLoss(logits, labels, inverse_normed_freqs):
    labels = labels.type(torch.float32)
    m = nn.Softmax(dim=1)
    probs = m(logits)
    yhat_for_gold = labels * probs
    yhat_for_gold_normalized = torch.sum(yhat_for_gold / labels.sum(dim=1).reshape((-1,1)),dim=1)
    logyhat_for_gold_normalized = torch.log(yhat_for_gold_normalized)
    return torch.mean(-logyhat_for_gold_normalized)

