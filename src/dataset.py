#imports
import os
import itertools as it
from collections import Counter
import spacy
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset

from PIL import Image

#run this if kernel dies after imshow
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#using spacy for the better text tokenization
spacy_eng = spacy.load("en_core_web_lg")


class Vocabulary:
    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}

        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}

        self.freq_threshold = freq_threshold

    def __len__(self): return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]


class FlickrDataset(Dataset):
    """
    FlickrDataset
    """

    df: pd.DataFrame

    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)#type:ignore
        self.transform = transform

        #Get image and caption colum from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        #Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())


    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")

        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)

        #numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)


class FlickrDataset_Plaintext(Dataset):
    """
    FlickrDataset
    Returns plaintext tokens (instead of using a vocabulary)
    """

    df: pd.DataFrame

    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)#type:ignore
        self.transform = transform

        #Get image and caption colum from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        #Initialize vocabulary and build vocab
        # self.vocab = Vocabulary(freq_threshold)
        # self.vocab.build_vocab(self.captions.tolist())


    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")

        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)

        #tokenize the caption text
        tokens= [*Vocabulary.tokenize(caption) , "<EOS>"] #we don’t need a <SOS> token with LSTM
        #(the initialization of the LSTM hidden vector suffices)

        # caption_vec += [self.vocab.stoi["<SOS>"]]
        # caption_vec += self.vocab.numericalize(caption)
        # caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, tokens

    
    
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

        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs, targets

def pad_embedding(caption: torch.Tensor, pad_token: torch.Tensor, size):
    caption_length, _ = caption.shape
    if caption_length > size:
        return caption[:size, :]
    return torch.cat([caption, *it.repeat(pad_token, size-caption_length)], axis=0) #type: ignore

class EmbedCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """

    def __init__(self,pad_token, embed):
        self.pad_token = pad_token
        self.embed = embed
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        captions = [self.embed(item[1]) for item in batch]
        max_len = max(len(c) for c in captions)
        padded = torch.stack(
            [pad_embedding(c, self.pad_token, size=max_len) for c in captions],
            dim=0
        )

        return imgs, padded
