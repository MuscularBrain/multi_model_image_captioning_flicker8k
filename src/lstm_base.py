import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        resnet = models.resnet152(pretrained = True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # resnet without the classification fc layer and batchnorm layer
        self.convs =  nn.Sequential(*list(resnet.children())[:-1])
        
        # resnet to embedding size
        self.linear = nn.Linear(resnet.fc.in_features, embedding_size)
        # batch norm the output
        self.bn = nn.BatchNorm1d(embedding_size)
        
        # initialize the weights
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        

    def forward(self, images):
        conv_out = self.convs(images)
        
        # reduce to 1d
        features = Variable(conv_out.data)
        features = features.view(features.size(0), -1)
        out = self.bn(self.linear(features))
        return out
    
    
    
MAX_CAPTION_LENGTH = 20

class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab, n_layers = 2):
        super(Decoder, self).__init__()        
        self.vocab = vocab
        self.embedding = nn.Embedding(len(self.vocab),embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(self.vocab))

        
        
        #self.embedding.weight.data.uniform_(-0.1, 0.1)
        #self.linear.weight.data.uniform_(-0.1, 0.1)
        #self.linear.bias.data.fill_(0)
    


        
    def forward(self, x, captions = None):
        """Decode image feature vectors and generates captions."""
        
        # training 
        if captions is not None:
            embeddings = self.embedding(captions)

            # teacher forcing
            inputs = torch.cat((x.unsqueeze(1), embeddings), 1)
            hiddens, _ = self.lstm(inputs)
            outputs = self.linear(hiddens[0])
            return outputs
        
        # testing/output:
        else:
            caption_out = []
            inputs = x.unsqueeze(1)
            states = None
            # recurrently generated caption words from previous state
            for i in range(MAX_CAPTION_LENGTH):                
                hiddens, states = self.lstm(inputs, states)        # (batch_size, 1, hidden_size), 
                outputs = self.linear(hiddens.squeeze(1))          # (batch_size, vocab_size)
                
                # sample output
                pred = outputs.max(1)[1]
                caption_out.append(pred)
                
                inputs = self.embedding(pred)
                inputs = inputs.unsqueeze(1)                       # (batch_size, 1, embedding_size)
                
                
            return torch.Tensor(caption_out)