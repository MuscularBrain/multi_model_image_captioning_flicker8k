import dataset
import torch
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset


#from https://www.kaggle.com/code/mdteach/torch-data-loader-flicker-8k/notebook

data_location = "../dataset"

#define the transform to be applied
transforms = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])


class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        targets = [item[1] for item in batch]
        
        lengths = [len(cap) for cap in targets]
        
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs, targets, lengths
    
if __name__ == "__main__":  
    flickr_dataset =  dataset.FlickrDataset(
        root_dir = data_location+"/Images",
        captions_file = data_location+"/captions.txt",
        transform=transforms
    )


    BATCH_SIZE = 4
    NUM_WORKER = 1

    #token to represent the padding
    pad_idx = flickr_dataset.vocab.stoi["<PAD>"]

    data_loader = DataLoader(
        dataset=flickr_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True,
        collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
    )


    for img, caps in data_loader:
        print(caps)
        break