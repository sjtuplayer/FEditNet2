"""Build the dataloader and sampler for training."""
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from .single_imglatent_dataset import ImgLatentDataset_ht,ImgLatentDataset_ht2,ImgLatentDataset_ht3



def build_dataloader(opt, seed, shuffle=True):
    dataset = ImgLatentDataset_ht(opt)
    print(f'# dataset: {len(dataset.img_paths)}')

    g = torch.Generator()
    g.manual_seed(seed)
    if shuffle:
        _sampler = RandomSampler(dataset, generator=g)
    else:
        _sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            sampler=_sampler, shuffle=False, drop_last=True)
    return dataloader

def build_dataloader2(opt, seed, shuffle=True):
    dataset = ImgLatentDataset_ht2(opt)
    print(f'# dataset: {len(dataset.img_paths)}')

    g = torch.Generator()
    g.manual_seed(seed)
    if shuffle:
        _sampler = RandomSampler(dataset, generator=g)
    else:
        _sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            sampler=_sampler, shuffle=False, drop_last=True)
    return dataloader

def build_dataloader3(opt, seed, shuffle=True):
    dataset = ImgLatentDataset_ht3(opt)
    print(f'# dataset: {len(dataset.img_paths)}')

    g = torch.Generator()
    g.manual_seed(seed)
    if shuffle:
        _sampler = RandomSampler(dataset, generator=g)
    else:
        _sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                            sampler=_sampler, shuffle=False, drop_last=True)
    return dataloader

def sample_data(dataloader):
    while True:
        for batch in dataloader:
            yield batch
