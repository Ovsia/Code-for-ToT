import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .utils.make_dataset_train import make_image_text1, make_image_text2, make_image_text3


class ToT_Typo(Dataset):
    def __init__(self, root, split='train', preprocess=None):

        self._data_dir = Path(root) / split
        self._preprocessed_dir = Path(root) / split
        self._typographic_dir = Path(root) / 'not_corr'/ 'typo_large' / split
        self.transform = preprocess

        with open(Path(root) / 'Labels_train.json') as f:
            id_to_class = json.load(f)

        self.id_to_class = id_to_class
        classes = []
        for dir in self._data_dir.iterdir():
            if not dir.is_dir():
                continue
            id = str(dir).split('/')[-1]
            class_i = id_to_class[id]
            classes.append(class_i)

        self.classes = classes
        self.class_to_idx = dict(zip(classes, range(len(classes))))
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

        self._labels = []

        files = list(self._data_dir.rglob('*'))
        self._files = []
        for i in range(len(files)):
            if not files[i].is_file():
                continue
            self._files.append(files[i])

        for file in self._files:
            id = str(file).split('/')[-2]
            class_i = id_to_class[id].split(',')[0]
            self._labels.append(self.class_to_idx[class_i])

        self._samples = []
        for file in self._files:
            typographic_path = self._typographic_dir / file.relative_to(self._data_dir)
            self._samples.append(typographic_path)

        if split=='train':
            with open('/ToT/attack_labels_train.json') as f:
                self.attack_labels = json.load(f)
            with open('/ToT/attack_texts_train.json') as f:
                self.attack_texts = json.load(f)

        if split=='val':
            with open('/ToT/attack_labels_val.json') as f:
                self.attack_labels = json.load(f)
            with open('/ToT/attack_texts_val.json') as f:
                self.attack_texts = json.load(f)

        if split=='test':
            with open('/ToT/attack_labels_test.json') as f:
                self.attack_labels = json.load(f)
            with open('/ToT/attack_texts_test.json') as f:
                self.attack_texts = json.load(f)


    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        image = self._samples[idx]
        label = self._labels[idx]
        image = Image.open(image)
        catogery = self.idx_to_class[self._labels[idx]]
        id = "_".join([str(self._files[idx]).split('/')[-2], str(self._files[idx]).split('/')[-1]])
        if self.transform is not None:
            image = self.transform(image)
            image_description = f"A photo of {catogery}."

        attack_text = self.attack_texts[idx]
        attack_label = self.attack_labels[idx]
        return image, label, catogery, image_description,id,attack_text,attack_label

    def _check_exists_synthesized_dataset(self) -> bool:
        return self._typographic_dir.is_dir()

