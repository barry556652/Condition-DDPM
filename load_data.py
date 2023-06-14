import random
from types import SimpleNamespace
from datasets import load_dataset, Image
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch
import torch.nn as nn


cfg = SimpleNamespace(    
    dataset_name = "barry556652/F1210_L2016_good_5000",
    image_column = "image",
    caption_column = "text",
    max_train_samples = 1,
    seed = 42,
    random_flip = "store_true",
    train_batch_size = 16,
    dataloader_num_workers = 0,
)
dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

def get_phison():
    
    if cfg.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            cfg.dataset_name,
        )
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(cfg.dataset_name, None)

    image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]

    caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    
    train_transforms = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
#             #centercrop越小速度越快
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    def class_label_tensor(examples, is_train=True):
        
        def class_tokenizer(text):
            class_names = ['C0201', 'R0201', 'L2016', 'F1210']
            class_label = text 
            num_classes = len(class_names)
            class_vector = torch.zeros(num_classes, dtype=torch.float)
            class_index = class_names.index(class_label)
            class_vector[class_index] = 1
            class_tensor = class_vector.view(1, num_classes)
            return torch.unsqueeze(torch.tensor(class_index), 0).type(torch.float)
        
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        label_tensor = class_tokenizer(captions[0])
        return label_tensor.type(torch.float)
    
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["class_label"] = class_label_tensor(examples)
        return examples
    
    train_dataset = dataset["train"].with_transform(preprocess_train)
    
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        class_label = torch.stack([example["class_label"] for example in examples])
        return {"pixel_values": pixel_values, "class_label": class_label}
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )
    
    return train_dataloader