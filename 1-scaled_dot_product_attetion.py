import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import math
import os
import urllib.request
from functools import partial
from urllib.error import HTTPError

def scaled_dot_product(q, k, v, mask=None):
    # implemented by the student, you can ignore the mask implementation currently
    # just assignment all the mask is on

    # Score Matric = QK^T 
    # Scaling is done: Scaled Score Matric = Score Matric / root d_k
    # But in this example we are comparing 2 words, thus there are q,k,v 

    score_matrix = q@k.mT
    scaled_score_matrix = score_matrix/ math.sqrt(q.shape[-1])
    attention_weight = torch.nn.functional.softmax(scaled_score_matrix)
    # output = attention_weight * v
    output = attention_weight@v
    return output, attention_weight

if __name__ == '__main__':
    # Test case
    # seq_len, d_k = 3, 2
    # torch.manual_seed(3025)
    # q = torch.randn(seq_len, d_k)
    # k = torch.randn(seq_len, d_k)
    # v = torch.randn(seq_len, d_k)
    # valid = torch.tensor([[-1.0142, -1.9154],
    #         [-0.4535, -1.6679],
    #         [ 0.5474, -1.2476]])
    # output, attention_weight = scaled_dot_product(q,k,v)
    # differences = (output - valid).mean()
    # print(q)
    # print(k)
    # print(v)
    # print(output)
    # print(differences)
    # assert torch.abs(differences) < 0.0001, 'the product must be similar output as expected'

    torch.set_printoptions(precision=16)
    q = torch.tensor([[-1.0590970516204834, 0.7565467357635498, 0.20126891136169434, -0.6364921927452087], [0.004584392067044973, -2.307931900024414, 1.4653981924057007, 0.8097707629203796], [0.4444141685962677, -1.1070520877838135, -0.4160226285457611, 1.122387170791626]])
    k = torch.tensor([[-13.375662803649902, 8.191596031188965, -32.305240631103516, 44.86949157714844], [15.18038272857666, -61.08460998535156, -41.45825958251953, -32.916629791259766], [48.51971435546875, 32.51784133911133, 1.7294328212738037, -62.1268310546875]])
    v = torch.tensor([[-36.20771408081055, 26.2475643157959, 8.842000961303711, 43.27354431152344], [-62.30213165283203, 41.48006820678711, -6.880943775177002, -21.451093673706055], [-5.928592205047607, -0.6987577080726624, -12.249312400817871, 48.32103729248047]])
    output, attention_weight = scaled_dot_product(q,k,v)

    print(attention_weight)
    # print(output)
