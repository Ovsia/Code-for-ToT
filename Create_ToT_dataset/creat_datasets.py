import argparse
import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.ToT import ToT
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():

    dataset = ToT(root='/ToT', split='train', preprocess=preprocess)
    dataset.make_typographic_attack_dataset()
    dataset.make_consistent_attack_dataset()
    dataset.make_nonsense_attack_dataset()

if __name__ == '__main__':
    main()