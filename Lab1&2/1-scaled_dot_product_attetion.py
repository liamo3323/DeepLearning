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
    q = torch.tensor([[3.3120908737182617, 11.714212417602539, -3.5947165489196777, -5.550578594207764], [9.422024726867676, 6.105061054229736, -2.6055192947387695, -4.635824680328369], [-8.967622756958008, 1.748089075088501, 4.3646955490112305, 2.2075018882751465]])
    k = torch.tensor([[49.388126373291016, -3.505145788192749, 6.966372013092041, 20.567304611206055], [9.32462215423584, 30.148683547973633, -1.962703824043274, -9.71722412109375], [-96.27658081054688, 24.82595443725586, 108.67559051513672, -20.08876609802246]])
    v = torch.tensor([[-25.174936294555664, 9.584662437438965, -25.932655334472656, -26.213214874267578], [17.62385368347168, -25.324230194091797, -35.68978500366211, -26.302963256835938], [12.274901390075684, 16.87058448791504, -14.158609390258789, 8.233379364013672]]) 
    output, attention_weight = scaled_dot_product(q,k,v)

    print(output)
    # print(output)
