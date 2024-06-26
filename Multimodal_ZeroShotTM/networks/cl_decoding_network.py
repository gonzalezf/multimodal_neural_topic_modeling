import torch
from torch import nn
from torch.nn import functional as F

from ctm_multimodal_package.networks.inference_network import CombinedInferenceNetwork, ContextualInferenceNetwork
# ------- New constrastive decoder network - felipe -------

#For now, let's keep the two inference networks! But the data will be the same.
#Then, we will use one network for BOW and the other for images!
#Num language = 2, means two inference networks
class ContrastiveDecoderNetworkFelipe(nn.Module):

    def __init__(self, input_size, bert_size, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True, label_size=0, num_languages=2):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            learn_priors : bool, make priors learnable parameter
            num_languages: no. of languages in dataset
        """
        super(ContrastiveDecoderNetworkFelipe, self).__init__()
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

        # input_size: same as vocab size
        self.input_size = input_size
        # n_components: no. of topics
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors

        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)

        # init prior parameters
        # \mu_1k = log \alpha_k + 1/K \sum_i log \alpha_i;
        # \alpha = 1 \forall \alpha
        # prior_mu is same for all languages
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        # prior_var is same for all languages
        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.num_languages = num_languages
        # each language has their own inference network (assume num_lang=2 for now)
        self.inf_net1 = ContextualInferenceNetwork(input_size, bert_size, n_components, hidden_sizes, activation)
        self.inf_net2 = ContextualInferenceNetwork(input_size, bert_size, n_components, hidden_sizes, activation)

        # topic_word_matrix is K x V, where L = no. of languages
        self.topic_word_matrix = None

        # beta is L x K x V where L = no. of languages
        self.beta = torch.Tensor(num_languages, n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)

        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, x_bert, labels=None):
        """Forward pass."""
        # x_bert: batch_size x L x bert_dim
        # print('DecoderNet - forward')
        # print('x_bert:', x_bert.shape)
        # pass to first x_bert to inference net1 (input is batch_size x bert_dim)
        
        #posterior_mu1, posterior_log_sigma1 = self.inf_net1(x[:, 0, :], x_bert[:, 0, :])
        posterior_mu1, posterior_log_sigma1 = self.inf_net1(x, x_bert)
        posterior_sigma1 = torch.exp(posterior_log_sigma1)

        # pass to second x_bert to inference net2 (input is batch_size x bert_dim)
        #posterior_mu2, posterior_log_sigma2 = self.inf_net2(x[:, 1, :], x_bert[:, 1, :])
        
        posterior_mu2, posterior_log_sigma2 = self.inf_net2(x, x_bert) #Here, the idea is to pass different data.
        posterior_sigma2 = torch.exp(posterior_log_sigma2)

        # generate separate thetas for each language
        z1 = self.reparameterize(posterior_mu1, posterior_log_sigma1)
        z2 = self.reparameterize(posterior_mu2, posterior_log_sigma2)
        theta1 = F.softmax(z1, dim=1)
        theta2 = F.softmax(z2, dim=1)
        # print("mu1:", posterior_mu1)
        # print("log_sigma1:", posterior_log_sigma1)
        # # print("z1:", z1)
        # print("-"*10)
        # print("mu2:", posterior_mu2)
        # print("log_sigma2:", posterior_log_sigma2)
        # # print("z2:", z2)
        # print("-" * 10)

        thetas_no_drop = torch.stack([theta1, theta2])
        z_no_drop = torch.stack([z1, z2])

        theta1 = self.drop_theta(theta1)
        theta2 = self.drop_theta(theta2)

        thetas = torch.stack([theta1, theta2])

        word_dist_collect = []
        for l in range(self.num_languages):
            # compute topic-word dist for each language
            # separate thetas and betas per language
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(thetas[l], self.beta[l])), dim=1)
            word_dist_collect.append(word_dist)

        # word_dist_collect: L x batch_size x input_size
        word_dist_collect = torch.stack([w for w in word_dist_collect])

        # topic_word_matrix and beta should be L x n_components x vocab_size
        self.topic_word_matrix = self.beta

        return self.prior_mean, self.prior_variance, posterior_mu1, posterior_sigma1, posterior_log_sigma1, \
            posterior_mu2, posterior_sigma2, posterior_log_sigma2, word_dist_collect, thetas_no_drop, z_no_drop

    def get_theta(self, x, x_bert, lang_index=0):
        with torch.no_grad():
            # we do inference PER LANGUAGE, so we use only 1 inference network at a time
            if lang_index == 0:
                posterior_mu, posterior_log_sigma = self.inf_net1(x, x_bert)
            else:
                posterior_mu, posterior_log_sigma = self.inf_net2(x, x_bert)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
            return theta