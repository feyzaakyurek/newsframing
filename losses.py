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

def FocalLoss2(logits, labels, inverse_normed_freqs):
    labels = labels.type(torch.float32)
    probs = torch.sigmoid(logits)
    pt = (1 - labels) * (1 - probs) + labels * probs
    log_pt = torch.log(pt)
    floss = - (1 - pt)**2 * log_pt
    alpha = inverse_normed_freqs.repeat(labels.shape[0]).view((labels.shape[0],-1))
    weights = (1 - labels) * (1 - alpha) + labels * alpha    
    floss_weighted =  floss * weights
    return torch.mean(floss_weighted)

def FocalLoss3(logits, labels, weights_0):
    batch_size = labels.shape[0]
    num_labels = labels.shape[1] # should be 9
    weights_1 = 1/num_labels - weights_0 
    labels = labels.type(torch.float32)
    probs = torch.sigmoid(logits)
    pt = (1 - labels) * (1 - probs) + labels * probs
    log_pt = torch.log(pt)
    floss = - (1 - pt)**2 * log_pt
    alpha_0 = weights_0.repeat(batch_size).view((batch_size,-1))
    alpha_1 = weights_1.repeat(batch_size).view((batch_size,-1))
    weights = (1 - labels) * alpha_0 + labels * alpha_1
    floss_weighted =  floss * weights
    return torch.mean(floss_weighted)


def BCELoss(logits, labels, inverse_normed_freqs=None):
    loss_fct = nn.BCEWithLogitsLoss()
    num_labels = labels.shape[1]
#     loss = loss_fct(logits.view(-1, num_labels).double(), labels.view(-1, self.num_labels).double())
    loss = loss_fct(logits.double(), labels.double())
    return loss

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

