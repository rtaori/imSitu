import math
from collections import Counter
import random


def initLinear(linear, val=None):
    if val is None:
        fan = linear.in_features + linear.out_features
        spread = math.sqrt(2.0) * math.sqrt(2.0 / fan)
    else:
        spread = val
    linear.weight.data.uniform_(-spread, spread)
    linear.bias.data.uniform_(-spread, spread)


def collapse_annotations(dataset, use_majority=True):
    dataset_collapsed = {}
    for image, annotations in dataset.items():
        frame_collapsed = {k: [] for k in annotations['frames'][0].keys()}
        for frame in annotations['frames']:
            for k, v in frame.items():
                if v.strip():
                    frame_collapsed[k].append(v)
        for k in frame_collapsed.keys():
            if len(frame_collapsed[k]) == 0:
                frame_collapsed[k] = ['']
            maj_v, count = Counter(frame_collapsed[k]).most_common(1)[0]
            if use_majority and count > 1:
                frame_collapsed[k] = maj_v
            else:
                frame_collapsed[k] = random.choice(frame_collapsed[k])
        dataset_collapsed[image] = {'verb': annotations['verb'], 'frames': [frame_collapsed]}
    return dataset_collapsed
