import os
import argparse
import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import dataset
from torch import nn, optim
import random
from datasets.ToT_origin import ToT_Origin
from datasets.ToT_nons import ToT_Nons
from datasets.ToT_consistent import ToT_Consistent
from datasets.ToT_typo import ToT_Typo

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.requires_grad:
            p.grad.data = p.grad.data.float()


def train( model,Origin,Nons,Consistent,Typo):


    Origin = DataLoader(Origin, batch_size=512,shuffle=True, num_workers=2)
    Nons = DataLoader(Nons, batch_size=512, shuffle=True, num_workers=2)
    Consistent = DataLoader(Consistent, batch_size=512, shuffle=True, num_workers=2)
    Typo = DataLoader(Typo, batch_size=512, shuffle=True, num_workers=2)
    batch_size = 512*4
    num_epochs = 5
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.visual.transformer.resblocks[-1].parameters():
        param.requires_grad = True

    model.visual.ln_post.weight.requires_grad=True
    model.visual.ln_post.bias.requires_grad=True


    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for i,(data, data_consistent,data_typo,data_nonsense) in enumerate(zip(Origin,Consistent,Typo,Nons)):
            image, label, catogery, image_description, id = data
            img = image.to(device)
            consistent_image, label, catogery_consistent, image_description, id = data_consistent
            consistent_words_img = consistent_image.to(device)

            typographic_image, label, catogery_typo, image_description, id,attack_text,attack_label = data_typo
            typo_img =  typographic_image.to(device)

            nonsense_image, label, catogery_nonsense, image_description, id = data_nonsense
            nonsense_words_img = nonsense_image.to(device)

            tensors = [img, consistent_words_img,typo_img, nonsense_words_img]

            img = torch.cat(tensors, dim=0)
            actual_batch_size = label.size(0)
            if actual_batch_size*4 < batch_size:
                break


            text1 = ["a photo of a" + catogery[i] for i in range(len(catogery))]
            text2 = ["a photo of a word written over a picture of a " + catogery_consistent[i] for i in range(len(catogery_consistent))]
            text3 = ["a photo of a word written over a picture of a " + catogery_typo[i] for i in range(len(catogery_typo))]
            text4 = ["a photo of a word written over a picture of a " + catogery_nonsense[i] for i in range(len(catogery_nonsense))]


            lists = [text1,text2,text3,text4]
            text = []
            [text.extend(lst) for lst in lists]



            text = clip.tokenize(text).to(device)

            logits_per_image, logits_per_text = model(img, text)

            if device == "cpu":
                ground_truth = torch.arange(batch_size).long().to(device)
            else:
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            optimizer.zero_grad()
            total_loss.backward()
            if device == "cpu":
                optimizer.step()

            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            print('[Epoch %d, Batch %d] loss: %.3f' % (epoch + 1, i + 1, total_loss))

        torch.save(model.state_dict(), f'/path_to_models/ours_epoch_{epoch + 1}.pt')

    return model

def main():
    model, preprocess = clip.load("ViT-B/32", device=device)

    Origin = ToT_Origin(root='/ToT',split='train',preprocess=preprocess)
    Consistent = ToT_Consistent(root='/ToT', split='train', preprocess=preprocess)
    Nons = ToT_Nons(root='/ToT', split='train', preprocess=preprocess)
    Typo = ToT_Typo(root='/ToT', split='train', preprocess=preprocess)

    train(model, Origin,Nons,Consistent,Typo)


if __name__ == '__main__':
    main()