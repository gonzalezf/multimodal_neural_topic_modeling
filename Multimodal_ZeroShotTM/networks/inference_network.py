from collections import OrderedDict
from torch import nn
import torch

#ZeroShotTM
class ContextualInferenceNetwork(nn.Module):

    """Inference Network."""

    def __init__(self, input_size, bert_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2, label_size=0, img_enc_dim=0): 
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input 
            output_size : int, dimension of output  #Outputsize is going to be the n_components
            hidden_sizes : tuple, length = n_layers #default (100,100)
            activation : string, 'softplus' or 'relu', default 'softplus' 
            dropout : float, default 0.2, default 0.2
        """
        super(ContextualInferenceNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(output_size, int), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        #Here, we receive the contextualized embeddings (we can concanetate the image embeddings)
        #self.input_layer = nn.Linear(bert_size + label_size, hidden_sizes[0])
        self.input_layer = nn.Linear(bert_size + label_size+img_enc_dim, hidden_sizes[0]) #incorporating images into CTM
        
        #self.adapt_bert = nn.Linear(bert_size, hidden_sizes[0])

        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x = None, x_bert =None, labels=None, X_image_embeddings = None):
        """Forward pass."""

        x = x_bert #i think they do this, because we dont care about BoW anymore. 
        if x_bert != None:
            if labels: #The current code, to support labels we need x_bert not be None
                x = torch.cat((x_bert, labels), 1)
        if X_image_embeddings != None:
            if x != None:
                #if x.shape[0]>1:#if x.shape[0] == X_image_embeddings.shape[0]:
                
                x = torch.cat((x, X_image_embeddings), 1)

            else:
                x = X_image_embeddings
            
        
        x = self.input_layer(x)
        
        x = self.activation(x)
        
        x = self.hiddens(x)
        
        x = self.dropout_enc(x)
        
        mu = self.f_mu_batchnorm(self.f_mu(x))
        
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
        
        return mu, log_sigma #These are the learned parameters
    #THIS IS OLD ! i AM NOT USING THIS FORWARD ANYMORE
    def forward_multimodal_old(self, x, x_bert, labels=None, X_image_embeddings = None):
        """Forward pass."""

        x = x_bert
        if labels:
            x = torch.cat((x_bert, labels), 1)
        if X_image_embeddings != None:
            
            x = torch.cat((x, X_image_embeddings), 1)
            
        
        x = self.input_layer(x)
        
        x = self.activation(x)
        
        x = self.hiddens(x)
        
        x = self.dropout_enc(x)
        
        mu = self.f_mu_batchnorm(self.f_mu(x))
        
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
        
        return mu, log_sigma #These are the learned parameters

#CombinedTM
class CombinedInferenceNetwork(nn.Module):

    """Inference Network."""

    def __init__(self, input_size, bert_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2, label_size=0,  img_enc_dim=0):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            output_size : int, dimension of output
            hidden_sizes : tuple, length = n_layers
            activation : string, 'softplus' or 'relu', default 'softplus'
            dropout : float, default 0.2, default 0.2
        """
        super(CombinedInferenceNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(output_size, int), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()


        self.adapt_bert = nn.Linear(bert_size, input_size)
        #self.bert_layer = nn.Linear(hidden_sizes[0], hidden_sizes[0])
        self.input_layer = nn.Linear(input_size + input_size + label_size + img_enc_dim, hidden_sizes[0]) #incorporating images into combinedtm

        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x, x_bert, labels=None, X_image_embeddings = None):
        """Forward pass."""
        x_bert = self.adapt_bert(x_bert) #adapt the bert vectors to the size of the input of BoW

        x = torch.cat((x, x_bert), 1) #my understanding, we concatenate BoW with the adaptation of xbert (after passing the vectors for a hidden layer)

        if labels is not None:
            x = torch.cat((x, labels), 1)

        if X_image_embeddings is not None:
            
            x = torch.cat((x, X_image_embeddings), 1)

        x = self.input_layer(x)

        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma
