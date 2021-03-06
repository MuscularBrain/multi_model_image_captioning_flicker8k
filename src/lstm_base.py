import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

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
        #print('Conv out shape')
        #print(features.size())
        out = self.bn(self.linear(features))
        return out
    
    
    
MAX_CAPTION_LENGTH = 20

class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab, n_layers = 2):
        super(Decoder, self).__init__()        
        self.vocab = vocab
        self.n_layers = n_layers
        self.hidden_size  = hidden_size
        self.embedding = nn.Embedding(len(self.vocab),embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(self.vocab))
        
        self.word_softmax = nn.Softmax(dim=-1)
        self.temp = 0.9

        
        
        #self.embedding.weight.data.uniform_(-0.1, 0.1)
        #self.linear.weight.data.uniform_(-0.1, 0.1)
        #self.linear.bias.data.fill_(0)
    
    
    
        
    def forward(self, x, captions = None):#, lengths = None):
        """Decode image feature vectors and generates captions."""
        
        # training 
        if captions is not None:
            captions = captions[:,:-1]
            embeddings = self.embedding(captions)

            # teacher forcing
            inputs = torch.cat((x.unsqueeze(1), embeddings), 1)
            #inputs = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False) 
                   
            outputs, _ = self.lstm(inputs)
            outputs = self.linear(outputs)
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
                # stochastic sampling with temperature softmax 
                out_soft = self.word_softmax(outputs)/self.temp
                pred = torch.multinomial(out_soft[0], 1)
                #pred = outputs.max(1)[1]
                if pred[0].item() == 0:
                    break
                caption_out.append(pred)
                
                
                inputs = self.embedding(pred)
                inputs = inputs.unsqueeze(1)                       # (batch_size, 1, embedding_size)
                
                
            return torch.Tensor(caption_out)

        
class BaseLSTM(nn.Module):      
    def __init__(self, embedding_size, hidden_size, vocab, n_layers = 2):
        super(BaseLSTM, self).__init__()
        self.encoder = Encoder(embedding_size)
        self.decoder = Decoder(embedding_size, hidden_size, vocab, n_layers)
    
    
    def forward(self, images, captions = None): #, lengths = None):
        encodings = self.encoder(images)
        return self.decoder(encodings, captions)#, lengths)