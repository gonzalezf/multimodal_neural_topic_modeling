import torch
from torch import nn
from torch.nn import functional as F

from Multimodal_ZeroShotTM.networks.inference_network import CombinedInferenceNetwork, ContextualInferenceNetwork


class DecoderNetworkMultimodal(nn.Module):
    #This is how the model can be used
    #self.model = DecoderNetwork(
            #bow_size, self.contextual_size, inference_type, n_components, model_type, hidden_sizes, activation,
            #dropout, learn_priors, label_size=label_size)


    def __init__(self, input_size, bert_size, infnet, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True, label_size=0, img_enc_dim = 0):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input #BoW_size
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            learn_priors : bool, make priors learnable parameter
        """
        super(DecoderNetworkMultimodal, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(n_components, int) and n_components > 0, \
            "n_components must be type int > 0."
        assert model_type in ['prodLDA', 'LDA'], \
            "model type must be 'prodLDA' or 'LDA'"
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size 
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.topic_word_matrix = None
        self.img_enc_dim = img_enc_dim
        
        
        if infnet == "zeroshot":
            print('ZeroShotTM THIS IS bert_size', bert_size)
            print('ZeroShotTM THIS IS input_size', input_size)
            self.inf_net = ContextualInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size, img_enc_dim=img_enc_dim)
        elif infnet == "combined":
            print('CombinedTM THIS IS bert_size', bert_size)
            print('CombinedTM THIS IS input_size', input_size)
            self.inf_net = CombinedInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size,  img_enc_dim=img_enc_dim)
        else:
            raise Exception('Missing infnet parameter, options are zeroshot and combined')

        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)

        # init prior parameters
        # \mu_1k = log \alpha_k + 1/K \sum_i log \alpha_i;
        # \alpha = 1 \forall \alpha
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = torch.Tensor(n_components, input_size)

        #Let's create a beta for the image
        self.beta_img = torch.Tensor(n_components, self.img_enc_dim) #Topic X Image_feature matrix (We will use to reconstruct image features)

        self.beta_text_features = torch.Tensor(n_components, bert_size) #Topic X Text_feature matrix (We will use to reconstruct text features)


        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
            self.beta_img = self.beta_img.cuda()
            self.beta_text_features = self.beta_text_features.cuda()


        self.beta = nn.Parameter(self.beta)
        self.beta_img = nn.Parameter(self.beta_img)
        self.beta_text_features = nn.Parameter(self.beta_text_features) 

        nn.init.xavier_uniform_(self.beta)
        
        nn.init.xavier_uniform_(self.beta_img) 
        nn.init.xavier_uniform_(self.beta_text_features) 

        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False) 
        #self.beta_img_batchnorm = nn.BatchNorm1d(img_enc_dim, affine=False) 
        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    #the encoder network generates the posterior mean (μ) and variance (σ^2), while the decoder network uses the prior mean (as a parameter) and the sampled latent representation to reconstruct the original data.
    
    
    def forward(self, x, x_bert, labels=None, X_image_embeddings = None,  txt_reconstruction = True, img_reconstruction = True):
        """Forward pass."""
        # batch_size x n_components
        #Inference network is part of the encoder
        #The decoder is part of the generative network.
        #The decoder takes the inferred latent variables and maps them back to the original dat
        
        # batch_size x n_components
        posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels, X_image_embeddings)
        posterior_sigma = torch.exp(posterior_log_sigma)
        
        #Theta are parameters of the distribution
        
        # generate samples from theta        
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        theta = self.drop_theta(theta)
        
        
        
        #Theta shape Bathsize X N_components (probably is the distribuition of topics per document)

        # prodLDA vs LDA
        
        
        # #Beta is shape: N_components X Bow_word_size
        #Beta is the latent space 
        
        if self.model_type == 'prodLDA':
            # in: batch_size x input_size x n_components
            
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1) 
            #thetha batchsize X N_components (topic distribution)
            #beta N_components x Bow_Size (topic-word weight matrix)
            # word_dist: batch_size x input_size #  The distribution of the vocab in the selection of documents (for reconstruction)
            if img_reconstruction:
                img_feature_dists = torch.matmul(theta, self.beta_img) 
            else:
                img_feature_dists = None
            if txt_reconstruction:
                predicted_textual_features =  torch.matmul(theta, self.beta_text_features)
            else:
                predicted_textual_features = None
            #img_feature_dists = self.beta_img_batchnorm(torch.matmul(theta, self.beta_img))
            #Batch_size X Img_feature_size
            
            
            
            self.topic_word_matrix = self.beta 
            if img_reconstruction:
                self.topic_img_feature_matrix = self.beta_img
            if txt_reconstruction:
                self.topic_text_feature_matrix = self.beta_text_features


        elif self.model_type == 'LDA':
            # simplex constrain on Beta
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            self.topic_word_matrix = beta
            word_dist = torch.matmul(theta, beta)
            # word_dist: batch_size x input_size
        else:
            raise NotImplementedError("Model Type Not Implemented")

        # classify labels

        estimated_labels = None

        if labels is not None:
            estimated_labels = self.label_classification(theta)

        return self.prior_mean, self.prior_variance, \
            posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, estimated_labels, img_feature_dists, predicted_textual_features

    def get_theta(self, x = None, x_bert = None, labels=None, X_image_embeddings=None):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels, X_image_embeddings)
            #posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta
