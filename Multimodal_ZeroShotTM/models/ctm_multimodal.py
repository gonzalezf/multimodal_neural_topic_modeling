import datetime
import multiprocessing as mp
import os
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import wordcloud
from scipy.special import softmax
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from Multimodal_ZeroShotTM.utils.early_stopping.early_stopping import EarlyStopping

#decoder network

from Multimodal_ZeroShotTM.networks.decoding_network import DecoderNetwork

from Multimodal_ZeroShotTM.networks.decoding_network_multimodal import DecoderNetworkMultimodal
# for contrastive loss
from pytorch_metric_learning.losses import NTXentLoss

import torch.nn as nn




class CTM_Multimodal:
    """Class to train the contextualized topic model. This is the more general class that we are keeping to
    avoid braking code, users should use the two subclasses ZeroShotTM and CombinedTm to do topic modeling.

    :param bow_size: int, dimension of input
    :param contextual_size: int, dimension of input that comes from BERT embeddings
    :param inference_type: string, you can choose between the contextual model and the combined model
    :param n_components: int, number of topic components, (default 10)
    :param model_type: string, 'prodLDA' or 'LDA' (default 'prodLDA')
    :param hidden_sizes: tuple, length = n_layers, (default (100, 100))
    :param activation: string, 'softplus', 'relu', (default 'softplus')
    :param dropout: float, dropout to use (default 0.2)
    :param learn_priors: bool, make priors a learnable parameter (default True)
    :param batch_size: int, size of batch to use for training (default 64)
    :param lr: float, learning rate to use for training (default 2e-3)
    :param momentum: float, momentum to use for training (default 0.99)
    :param solver: string, optimizer 'adam' or 'sgd' (default 'adam')
    :param num_epochs: int, number of epochs to train for, (default 100)
    :param reduce_on_plateau: bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
    :param num_data_loader_workers: int, number of data loader workers (default cpu_count). set it to 0 if you are using Windows
    :param label_size: int, number of total labels (default: 0)
    :param loss_weights: dict, it contains the name of the weight parameter (key) and the weight (value) for each loss.
    It supports only the weight parameter beta for now. If None, then the weights are set to 1 (default: None).

    """

    def __init__(self, bow_size, contextual_size, inference_type="combined", n_components=10, model_type='prodLDA',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2, learn_priors=True, batch_size=64,
                 lr=2e-3, momentum=0.99, solver='adam', num_epochs=100, reduce_on_plateau=False,
                 num_data_loader_workers=mp.cpu_count(), label_size=0, loss_weights=None, img_enc_dim = 0):

        self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        print('Initializing CTM  Multimodal class')
        if self.__class__.__name__ == "CTM":
            raise Exception("You cannot call this class. Use ZeroShotTM or CombinedTM")

        assert isinstance(bow_size, int) and bow_size > 0, \
            "input_size must by type int > 0."
        assert isinstance(n_components, int) and bow_size > 0, \
            "n_components must by type int > 0."
        assert model_type in ['LDA', 'prodLDA'], \
            "model must be 'LDA' or 'prodLDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."
        assert isinstance(learn_priors, bool), "learn_priors must be boolean."
        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size must be int > 0."
        assert lr > 0, "lr must be > 0."
        assert isinstance(momentum, float) and 0 < momentum <= 1, \
            "momentum must be 0 < float <= 1."
        assert solver in ['adam', 'sgd'], "solver must be 'adam' or 'sgd'."
        assert isinstance(reduce_on_plateau, bool), \
            "reduce_on_plateau must be type bool."
        assert isinstance(num_data_loader_workers, int) and num_data_loader_workers >= 0, \
            "num_data_loader_workers must by type int >= 0. set 0 if you are using windows"

        self.bow_size = bow_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.contextual_size = contextual_size
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.num_data_loader_workers = num_data_loader_workers
        self.training_doc_topic_distributions = None

        self.img_enc_dim = img_enc_dim

        if loss_weights:
            self.weights = loss_weights
        else:
            #self.weights = {"KL": 1, 'EL':50} #We will use 1 as the CTM paper #beta and KL is the same, for the CTM and M3L paper respectively . #self.weights = {"beta": 1}
            self.weights = {"KL": 1, "BoW_loss": 1, 'IMG_loss':1, 'TXT_loss':1} 
        
        print('Current loss weights: ', self.weights)

        #reconstruction of images

        self.model = DecoderNetworkMultimodal( 
            bow_size, self.contextual_size, inference_type, n_components, model_type, hidden_sizes, activation,
            dropout, learn_priors, label_size=label_size, img_enc_dim=img_enc_dim)
        
        #constrastive learning
        #self.model = ContrastiveDecoderNetworkFelipe(
            #bow_size, self.contextual_size, n_components, model_type, hidden_sizes, activation,
            #dropout, learn_priors, label_size=label_size)

        self.early_stopping = None

        # init optimizer
        if self.solver == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum)

        # init lr scheduler
        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)

        # performance attributes
        self.best_loss_train = float('inf')

        # training attributes
        self.model_dir = None
        self.nn_epoch = None

        # validation attributes
        self.validation_data = None

        # learned topics
        self.best_components = None
        self.best_components_img = None

        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

        self.model = self.model.to(self.device)

    
    def _infoNCE_loss(self, embeddings1, embeddings2, temperature=0.07):
        
        batch_size = embeddings1.shape[0]
        labels = torch.arange(batch_size)
        labels = torch.cat([labels, labels])
        embeddings_cat = torch.cat([embeddings1, embeddings2])
        loss_func = NTXentLoss()
        
        infonce_loss = loss_func(embeddings_cat, labels)
        
        return infonce_loss

    
    def _kl_loss1(self, thetas1, thetas2):
        theta_kld = F.kl_div(thetas1.log(), thetas2, reduction='sum')
        return theta_kld

    
    
    '''
    self._loss(X_bow, word_dists, prior_mean, prior_variance,
                posterior_mean, posterior_variance, posterior_log_variance, X_image_embeddings, img_feature_dists)
        
    '''
    
    def _loss(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance, input_img_features = None, predicted_img_features =None, X_textual_embeddings = None, predicted_textual_features = None,  img_reconstruction = True, txt_reconstruction = True ):
        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        KL = 0.5 * (
            var_division + diff_term - self.n_components + logvar_det_division)

        # Reconstruction term
        #RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)
        RL = self._rl_loss(inputs, word_dists)
        #RL_img = self._rl_loss_img(input_img_features,predicted_img_features )
        if img_reconstruction:
            #RL_img = self._rl_loss_img_l2(input_img_features,predicted_img_features )
            RL_img = self._rl_loss_img_cosine(input_img_features,predicted_img_features )
            RL_img = RL_img.reshape(RL_img.shape[0]) #this is new
        else:
            RL_img = None
        if txt_reconstruction:
            #RL_text = self._rl_loss_img_l2(X_textual_embeddings,predicted_textual_features )
            RL_text = self._rl_loss_img_cosine(X_textual_embeddings,predicted_textual_features )
            RL_text = RL_text.reshape(RL_text.shape[0]) #this is new
        else:
            RL_text = None        

        #loss = self.weights["beta"]*KL + RL

        return KL, RL, RL_img, RL_text

    #This is new. See if it works
    def _rl_loss(self, true_word_dists, pred_word_dists):
        # Reconstruction term
        
        RL = -torch.sum(true_word_dists * torch.log(pred_word_dists + 1e-10), dim=1)
        return RL
    def _rl_loss_img(self, true_word_dists, pred_word_dists):
        # Reconstruction term
        
        RL = -torch.sum(true_word_dists * torch.log(pred_word_dists + 1e-10), dim=1)
        return RL
    
    
    
    #RL = -torch.sum(true_word_dists * torch.log(pred_word_dists + 1e-10), dim=1)
    def _rl_loss_img_cosine(self, true_word_dists, pred_word_dists):
        # Reconstruction term
        # Concatenate the tensors along the first dimension (the batch dimension)
        
        
        target = torch.full((32,), 1).cuda()    
        loss_img = nn.CosineEmbeddingLoss(reduction='none')
        output = loss_img(true_word_dists, pred_word_dists, target)
        RL = output.reshape(output.shape[0], 1)
        

        # Compute the mean of the loss over the second dimension of the tensor
        
        #RL = torch.mean(output, dim=1, keepdim=True)

        
        return RL 
    def _rl_loss_img_l2(self, true_word_dists, pred_word_dists):
        
        # Reconstruction term
        # Concatenate the tensors along the first dimension (the batch dimension)
        


        loss_img = nn.MSELoss(reduction='none')
        output = loss_img(true_word_dists, pred_word_dists)
        

        # Compute the mean of the loss over the second dimension of the tensor
        #RL = output.reshape(output.shape[0], 1)
        RL = torch.mean(output, dim=1, keepdim=True)

        
        return RL 
    def _rl_loss_img_l2_old(self, true_word_dists, pred_word_dists):
        # Reconstruction term
        # Concatenate the tensors along the first dimension (the batch dimension)

        combined = torch.cat([true_word_dists, pred_word_dists], dim=0)

        # Compute the mean and standard deviation of the concatenated tensor
        mean = torch.mean(combined)
        std = torch.std(combined)

        # Normalize the tensors using the mean and standard deviation of the concatenated tensor
        true_word_dists = (true_word_dists - mean) / std
        pred_word_dists = (pred_word_dists - mean) / std

        loss_img = nn.MSELoss(reduction='none')
        output = loss_img(true_word_dists, pred_word_dists)
        print('SHAPE output', output.shape)

        # Compute the mean of the loss over the second dimension of the tensor
        RL = torch.mean(output, dim=1, keepdim=True)

        
        return RL 


    def _train_epoch(self, loader, multimodal=True, txt_reconstruction = False, img_reconstruction = True, epoch=None):
        """Train epoch."""
        self.model.train()
        samples_processed = 0
        train_loss = 0
        train_loss_rl_bow = 0
        train_loss_rl_img =0
        train_loss_rl_txt =0
        train_loss_kl = 0
        list_txt_weight_epoch_0 = []
        list_img_weight_epoch_0 = []
        for batch_samples in loader:
            # batch_size x vocab_size
            X_bow = batch_samples['X_bow']
            X_bow = X_bow.reshape(X_bow.shape[0], -1)
            
            X_contextual = batch_samples['X_contextual']
            #X_image_embeddings = batch_samples['X_image_embeddings']
            #X_image_embeddings_index = batch_samples['X_image_embeddings_index']
            


            if multimodal:
                X_image_embeddings = batch_samples['X_image_embeddings']
                X_image_embeddings_index = batch_samples['X_image_embeddings_index']

            if "labels" in batch_samples.keys():
                labels = batch_samples["labels"]
                labels = labels.reshape(labels.shape[0], -1)
                labels = labels.to(self.device)
            else:
                labels = None

            
            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()
                if multimodal:
                    X_image_embeddings = X_image_embeddings.cuda()
                    #X_image_embeddings_index = X_image_embeddings_index.cuda() #I did this
                    

            # forward pass
            self.model.zero_grad()


            prior_mean, prior_variance, posterior_mean, posterior_variance,\
            posterior_log_variance, word_dists, estimated_labels, img_feature_dists, predicted_textual_features = self.model(X_bow, X_contextual, labels, X_image_embeddings =  X_image_embeddings,  txt_reconstruction = True, img_reconstruction = True)



            kl_loss, rl_loss, rl_img_loss, rl_text_loss = self._loss(
                X_bow, word_dists, prior_mean, prior_variance,
                posterior_mean, posterior_variance, posterior_log_variance, X_image_embeddings, img_feature_dists, X_textual_embeddings = X_contextual, predicted_textual_features = predicted_textual_features,  img_reconstruction = img_reconstruction, txt_reconstruction = txt_reconstruction)

            
            loss = self.weights["BoW_loss"] * rl_loss + self.weights["KL"] * kl_loss
            
            if img_reconstruction:
                if epoch == 0  and self.weights["IMG_loss"] != -1:
                    self.final_weight_img_loss = self.weights["IMG_loss"] * (self.weights["BoW_loss"] * rl_loss.sum().item()/rl_img_loss.sum().item())
                    
                    list_img_weight_epoch_0.append(self.final_weight_img_loss)
                if self.weights["IMG_loss"] == -1:
                    self.final_weight_img_loss = 1

                #loss += self.weights["IMG_loss"] * rl_img_loss
                loss += self.final_weight_img_loss * rl_img_loss #this is new
            if txt_reconstruction:
                #train_loss_rl_txt += self.weights["TXT_loss"] * rl_text_loss.sum().item() #This was the standard
                if epoch ==0 and self.weights["TXT_loss"] != -1:
                    self.final_weight_txt_loss = self.weights["TXT_loss"] * (self.weights["BoW_loss"] * rl_loss.sum().item()/rl_text_loss.sum().item())
                    list_txt_weight_epoch_0.append(self.final_weight_txt_loss)
                #else:
                    #final_weight_txt_loss is the mean of the weights of the first epoch stored at list_txt_weight_epoch_0                    
                    
                if self.weights["TXT_loss"] == -1:  #if the parameter is -1, it means the cosine loss is equal to 1.0 and not 1x BowLOSS OR 2X BOWW LOSS etc
                    self.final_weight_txt_loss = 1


                #loss += self.weights["TXT_loss"] * rl_text_loss #
                loss += self.final_weight_txt_loss * rl_text_loss #this is new

            loss = loss.sum()
            loss.backward()
            self.optimizer.step()

            #compute train loss
            samples_processed += X_bow.size()[0]
            train_loss += loss.item()
            train_loss_rl_bow += self.weights["BoW_loss"] * rl_loss.sum().item()
            if img_reconstruction:
                #train_loss_rl_img += self.weights["IMG_loss"] * rl_img_loss.sum().item()
                train_loss_rl_img += self.final_weight_img_loss * rl_img_loss.sum().item()
            
            if txt_reconstruction:
                train_loss_rl_txt += self.final_weight_txt_loss  * rl_text_loss.sum().item() #Probando, la idea es dejar uno en funcion de lo otro
            
            train_loss_kl += self.weights["KL"]*kl_loss.sum().item()

        if epoch ==0 and self.weights["TXT_loss"] != -1:
            self.final_weight_txt_loss = np.mean(list_txt_weight_epoch_0)
            print('MEAN New weight for txt loss', self.final_weight_txt_loss)
        
        if epoch ==0 and self.weights["IMG_loss"] != -1:
            self.final_weight_img_loss = np.mean(list_img_weight_epoch_0)
            print('MEAN New weight for img loss', self.final_weight_img_loss)
        train_loss /= samples_processed
        train_loss_rl_bow /= samples_processed
        train_loss_rl_img /= samples_processed
        train_loss_kl /= samples_processed
        train_loss_rl_txt /= samples_processed
        
        return samples_processed, train_loss, train_loss_rl_bow, train_loss_rl_img, train_loss_kl, train_loss_rl_txt

    def fit(self, train_dataset, validation_dataset=None, save_dir=None, verbose=False, patience=5, delta=0,
            n_samples=20, multimodal=True, tensorboard = False,  txt_reconstruction = True, img_reconstruction = True):
        """
        Train the CTM model.

        :param train_dataset: PyTorch Dataset class for training data.
        :param validation_dataset: PyTorch Dataset class for validation data. If not None, the training stops if validation loss doesn't improve after a given patience
        :param save_dir: directory to save checkpoint models to.
        :param verbose: verbose
        :param patience: How long to wait after last time validation loss improved. Default: 5
        :param delta: Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        :param n_samples: int, number of samples of the document topic distribution (default: 20)

        """
        # Print settings to output file
        if verbose:
            print("Settings: \n\
                   N Components: {}\n\
                   Topic Prior Mean: {}\n\
                   Topic Prior Variance: {}\n\
                   Model Type: {}\n\
                   Hidden Sizes: {}\n\
                   Activation: {}\n\
                   Dropout: {}\n\
                   Learn Priors: {}\n\
                   Learning Rate: {}\n\
                   Momentum: {}\n\
                   Reduce On Plateau: {}\n\
                   Save Dir: {}".format(
                self.n_components, 0.0,
                1. - (1. / self.n_components), self.model_type,
                self.hidden_sizes, self.activation, self.dropout, self.learn_priors,
                self.lr, self.momentum, self.reduce_on_plateau, save_dir))

        self.model_dir = save_dir
        self.idx2token = train_dataset.idx2token
        train_data = train_dataset
        self.validation_data = validation_dataset
        if self.validation_data is not None:
            self.early_stopping = EarlyStopping(patience=patience, verbose=verbose, path=save_dir, delta=delta)
        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_data_loader_workers, drop_last=True)

        # init training variables
        train_loss = 0
        samples_processed = 0
        train_loss_rl_bow = 0
        train_loss_rl_img = 0


        # train loop
        pbar = tqdm(self.num_epochs, position=0, leave=True)
        if tensorboard:
            writer = SummaryWriter()


        for epoch in range(self.num_epochs):
            print("-"*10, "Epoch", epoch+1, "-"*10)
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss, train_loss_rl_bow, train_loss_rl_img, train_loss_kl, train_loss_rl_text = self._train_epoch(train_loader, multimodal=multimodal, txt_reconstruction = txt_reconstruction, img_reconstruction = img_reconstruction, epoch=epoch)
            #samples_processed, train_loss, train_loss_rl_bow, train_loss_rl_img, train_loss_kl, train_loss_rl_txt
            samples_processed += sp
            e = datetime.datetime.now()
            pbar.update(1)

            if self.validation_data is not None:
                validation_loader = DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=True,
                                               num_workers=self.num_data_loader_workers, drop_last=True)
                # train epoch
                s = datetime.datetime.now()
                val_samples_processed, val_loss = self._validation(validation_loader)
                e = datetime.datetime.now()

                # report
                if verbose:
                    print("Epoch: [{}/{}]\tSamples: [{}/{}]\tValidation Loss: {}\tTime: {}".format(
                        epoch + 1, self.num_epochs, val_samples_processed,
                        len(self.validation_data) * self.num_epochs, val_loss, e - s))

                pbar.set_description("Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tValid Loss: {}\tTime: {}".format(
                    epoch + 1, self.num_epochs, samples_processed,
                    len(train_data) * self.num_epochs, train_loss, val_loss, e - s))

                self.early_stopping(val_loss, self)
                if self.early_stopping.early_stop:
                    print("Early stopping")

                    break
            else:
                # save last epoch
                self.best_components = self.model.beta
                self.best_components_img = self.model.beta_img
                self.best_components_text = self.model.beta_text_features
                
                if save_dir is not None:
                    self.save(save_dir)
            pbar.set_description("Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tTime: {}\tTrain RL BOW Loss: {}\t\tTrain  RL IMG Loss: {}\tTrain  KL Loss: {}\tTrain  RL text: {}\t".format(
                epoch + 1, self.num_epochs, samples_processed,
                len(train_data) * self.num_epochs, train_loss, e - s,  train_loss_rl_bow, train_loss_rl_img, train_loss_kl, train_loss_rl_text))
            if tensorboard:
                writer.add_scalar('Loss/total', train_loss, epoch)
                writer.add_scalar('Loss/rl_bow', train_loss_rl_bow, epoch)
                writer.add_scalar('Loss/rl_img', train_loss_rl_img, epoch)
                writer.add_scalar('Loss/rl_text', train_loss_rl_text, epoch)
                writer.add_scalar('Loss/kl_divergence', train_loss_kl, epoch)



        pbar.close()
        self.training_doc_topic_distributions = self.get_doc_topic_distribution(train_dataset, n_samples)

    def _validation(self, loader):
        """Validation epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            # batch_size x vocab_size
            X_bow = batch_samples['X_bow']
            X_bow = X_bow.reshape(X_bow.shape[0], -1)
            X_contextual = batch_samples['X_contextual']

            if "labels" in batch_samples.keys():
                labels = batch_samples["labels"]
                labels = labels.to(self.device)
                labels = labels.reshape(labels.shape[0], -1)
            else:
                labels = None

            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()

            # forward pass
            self.model.zero_grad()
            prior_mean, prior_variance, posterior_mean, posterior_variance, posterior_log_variance, word_dists, \
            estimated_labels =\
                self.model(X_bow, X_contextual, labels)

            kl_loss, rl_loss = self._loss(X_bow, word_dists, prior_mean, prior_variance,
                              posterior_mean, posterior_variance, posterior_log_variance)

            loss = self.weights["beta"]*kl_loss + rl_loss
            loss = loss.sum()

            if labels is not None:
                target_labels = torch.argmax(labels, 1)
                label_loss = torch.nn.CrossEntropyLoss()(estimated_labels, target_labels)
                loss += label_loss

            # compute train loss
            samples_processed += X_bow.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss

    def get_thetas(self, dataset, n_samples=20):
        """
        Get the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via
        the parameter n_sample.

        :param dataset: a PyTorch Dataset containing the documents
        :param n_samples: the number of sample to collect to estimate the final distribution (the more the better).
        """
        return self.get_doc_topic_distribution(dataset, n_samples=n_samples)



    def get_doc_topic_distribution(self, dataset, n_samples=20):
        """
        Get the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via
        the parameter n_sample.

        :param dataset: a PyTorch Dataset containing the documents
        :param n_samples: the number of sample to collect to estimate the final distribution (the more the better).
        """
        self.model.eval()

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_data_loader_workers)
        pbar = tqdm(n_samples, position=0, leave=True)
        final_thetas = []
        for sample_index in range(n_samples):
            with torch.no_grad():
                collect_theta = []
                for batch_samples in loader:
                    # batch_size x vocab_size
                    if "X_bow" in batch_samples.keys():
                        X_bow = batch_samples['X_bow']
                        X_bow = X_bow.reshape(X_bow.shape[0], -1)
                        
                    else:
                        # Here we are creating a vector with the size of the vocabulary and we are filling it with zeros                    
                        X_bow = torch.zeros(batch_samples['X_image_embeddings'].shape[0], self.vocab_size)
                    X_bow = X_bow.to(self.device)
                    if "X_contextual" in batch_samples.keys():
                        X_contextual = batch_samples['X_contextual']
                    else:
                        #We must do this. We are assuming that we train the model in multimodal settings
                        #and then we  perform inference in both image and text, only text, or only images
                        #Here we are creating a vector with the size of the contextual size and we are filling it with zeros
                        X_contextual = torch.zeros(batch_samples['X_image_embeddings'].shape[0], self.contextual_size)
                    X_contextual = X_contextual.to(self.device)
                    if "X_image_embeddings" in batch_samples.keys():	
                        X_image_embeddings = batch_samples['X_image_embeddings']
                    else:
                        #Here we are creating a vector with the size of the image embedding size and we are filling it with zeros This is important for inference
                        X_image_embeddings = torch.zeros(batch_samples['X_contextual'].shape[0], self.img_enc_dim)
                    X_image_embeddings = X_image_embeddings.to(self.device)

                    if "X_image_embeddings_index" in batch_samples.keys():
                        X_image_embeddings_index = batch_samples['X_image_embeddings_index']
                        #X_image_embeddings_index = X_image_embeddings_index.to(self.device) #I modified this

                    
                    #batch_samples.keys() dict_keys(['X_bow', 'X_contextual', 'X_image_embeddings', 'X_image_embeddings_index'])
                    
                    if "labels" in batch_samples.keys():
                        labels = batch_samples["labels"]
                        labels = labels.to(self.device)
                        labels = labels.reshape(labels.shape[0], -1)
                    else:
                        labels = None

                  
                    # forward pass
                    self.model.zero_grad()
                    
                    collect_theta.extend(self.model.get_theta(X_bow, X_contextual, labels, X_image_embeddings).cpu().numpy().tolist())

                pbar.update(1)
                pbar.set_description("Sampling: [{}/{}]".format(sample_index + 1, n_samples))

                final_thetas.append(np.array(collect_theta))
        pbar.close()
        return np.sum(final_thetas, axis=0) / n_samples

    def get_most_likely_topic(self, doc_topic_distribution):
        """ get the most likely topic for each document

        :param doc_topic_distribution: ndarray representing the topic distribution of each document
        """
        return np.argmax(doc_topic_distribution, axis=0)


    def get_topics(self, k=10):##CTM ORIGINAL
        """
        Retrieve topic words.

        :param k: int, number of words to return per topic, default 10.
        """
        assert k <= self.bow_size, "k must be <= input size."
        component_dists = self.best_components
        topics = defaultdict(list)
        for i in range(self.n_components):
            _, idxs = torch.topk(component_dists[i], k)
            component_words = [self.idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            topics[i] = component_words
        return topics
    
    #Get closest image embeddings based on their similarity to beta img
    def get_topics_image_embeddings_beta_img(self,image_embeddings, image_embeddings_index, k=10):
        '''
        topics = defaultdict(list)
        for current_topic_id in range(self.n_components):
            top_vectors = self.get_list_of_closest_image_embeddings_to_beta_img(current_topic_id, image_embeddings, image_embeddings_index, k)
            topics[current_topic_id] = top_vectors
        return topics
        '''
        
        topics = []
        for current_topic_id in range(self.n_components):
            top_vectors = self.get_list_of_closest_image_embeddings_to_beta_img(current_topic_id, image_embeddings, image_embeddings_index, k)
            topics.append(top_vectors)
        return topics
        
    #Get top image embeddings per topic
    def get_list_of_closest_image_embeddings_to_beta_img(self, topic_id, image_embeddings, image_embeddings_index, k=10):#
        """
        Return the images most similar to the topic_image_embedding feature learned
        :param k: int, number of image_features to return per topic, default 10.

        """
        #representative vector for that topic
        representative_img_feature_vector = self.best_components_img[topic_id]

        #Cosine similarity returns value larger than 1 . https://github.com/pytorch/pytorch/issues/78064
        #https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb#scrollTo=yKAxkQR7bf3A Note that here they normalized the vectors.    
        #To avoid any problems I will use the function of cosine similarity included in torch.nn

        #New solution - good
        query_similarity = torch.nn.functional.cosine_similarity(representative_img_feature_vector.unsqueeze(0), image_embeddings, dim=1)
        
        #old solution - bad
        #query_similarity = representative_img_feature_vector @ image_embeddings.T #This is bad, because it returns values larger than 1. This is because I didnt normalize the vectors, that is why
        
        query_similarity = query_similarity.cpu().detach().numpy()
        
        list_closest_image_embeddings = []
        for i in range(1,k+1):
            image_index = np.argsort(query_similarity, axis=0)[-i]
            #current_similarity_score = query_similarity[image_index]
            #current_image_id = image_embeddings_index[image_index]
            current_image_embedding = image_embeddings[image_index]
            
            #list_imageId_similarity.append((current_image_id, current_similarity_score))
            #convert current_image_embedding to numpy
            #current_image_embedding = current_image_embedding.cpu().detach().numpy()

            list_closest_image_embeddings.append(current_image_embedding)
        
        return list_closest_image_embeddings



    
    def get_topics_3ML(self, k=10): #I used this when there were more than 1 inference network (e.g., crosslingual from 3ML)
        """
        Retrieve topic words.

        :param k: int, number of words to return per topic, default 10.
        """
        assert k <= self.bow_size, "k must be <= input size."
        component_dists = self.best_components 
        #This is considering two languages! We only need the first one
        component_dists = component_dists[0] #The index here (e.g., 0) indicates the inference network that I am trying to access
        topics = defaultdict(list)
        for i in range(self.n_components):
            
            _, idxs = torch.topk(component_dists[i], k)
            component_words = [self.idx2token[idx]
                                for idx in idxs.cpu().numpy()]
            topics[i] = component_words
        return topics

    def get_topics_multilingual(self, k=10): #DELETE M3L original MULTILINGUAL
        """
        Retrieve topic words.

        :param k: int, number of words to return per topic, default 10.
        """
        assert k <= self.bow_size, "k must be <= input size."
        component_dists = self.best_components
        
        topics_all = []
        for l in range(self.num_lang):
            topics = defaultdict(list)
            for i in range(self.n_components):
                _, idxs = torch.topk(component_dists[l][i], k)
                component_words = [self.train_data.idx2token[l][idx]
                                   for idx in idxs.cpu().numpy()]
                topics[i] = component_words
            topics_all.append(topics)
        return topics_all

    def get_topic_lists(self, k=10):
        """
        Retrieve the lists of topic words.
        :param k: (int) number of words to return per topic, default 10.
        """
        assert k <= self.bow_size, "k must be <= input size."
        
        component_dists = self.best_components
        topics = []
        for i in range(self.n_components):
            _, idxs = torch.topk(component_dists[i], k)
            component_words = [self.idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            topics.append(component_words)
        return topics


        
    def get_topic_lists_3ML(self, k=10):
        """
        Retrieve the lists of topic words.

        :param k: (int) number of words to return per topic, default 10.
        """
        assert k <= self.bow_size, "k must be <= input size."
        
        component_dists = self.best_components
        #This is considering two languages! We only need the first one
        component_dists = component_dists[0]
        
        topics = []
        for i in range(self.n_components):
            _, idxs = torch.topk(component_dists[i], k)
            component_words = [self.idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            topics.append(component_words)
        return topics

    def _format_file(self):
        model_dir = "contextualized_topic_model_nc_{}_tpm_{}_tpv_{}_hs_{}_ac_{}_do_{}_lr_{}_mo_{}_rp_{}". \
            format(self.n_components, 0.0, 1 - (1. / self.n_components),
                   self.model_type, self.hidden_sizes, self.activation,
                   self.dropout, self.lr, self.momentum,
                   self.reduce_on_plateau)
        return model_dir

    def save(self, models_dir=None):
        """
        Save model. (Experimental Feature, not tested)

        :param models_dir: path to directory for saving NN models.
        """
        warnings.simplefilter('always', Warning)
        warnings.warn("This is an experimental feature that we has not been fully tested. Refer to the following issue:"
                      "https://github.com/MilaNLProc/contextualized-topic-models/issues/38",
                      Warning)

        if (self.model is not None) and (models_dir is not None):

            model_dir = self._format_file()
            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)

    def load(self, model_dir, epoch):
        """
        Load a previously trained model. (Experimental Feature, not tested)

        :param model_dir: directory where models are saved.
        :param epoch: epoch of model to load.
        """

        warnings.simplefilter('always', Warning)
        warnings.warn("This is an experimental feature that we has not been fully tested. Refer to the following issue:"
                      "https://github.com/MilaNLProc/contextualized-topic-models/issues/38",
                      Warning)

        epoch_file = "epoch_" + str(epoch) + ".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            checkpoint = torch.load(model_dict, map_location=torch.device(self.device))

        for (k, v) in checkpoint['dcue_dict'].items():
            setattr(self, k, v)

        self.model.load_state_dict(checkpoint['state_dict'])

    def get_topic_word_matrix(self):
        """
        Return the topic-word matrix (dimensions: number of topics x length of the vocabulary).
        If model_type is LDA, the matrix is normalized; otherwise the matrix is unnormalized.
        """
        return self.model.topic_word_matrix.cpu().detach().numpy()

    def get_topic_word_distribution(self):
        """
        Return the topic-word distribution (dimensions: number of topics x length of the vocabulary).
        """
        mat = self.get_topic_word_matrix()
        return softmax(mat, axis=1)

    def get_word_distribution_by_topic_id(self, topic):
        """
        Return the word probability distribution of a topic sorted by probability.

        :param topic: id of the topic (int)

        :returns list of tuples (word, probability) sorted by the probability in descending order
        """
        if topic >= self.n_components:
            raise Exception('Topic id must be lower than the number of topics')
        else:
            wd = self.get_topic_word_distribution()
            t = [(word, wd[topic][idx]) for idx, word in self.idx2token.items()]
            t = sorted(t, key=lambda x: -x[1])
        return t

    def get_wordcloud(self, topic_id, n_words=5, background_color="black", width=1000, height=400):
        """
        Plotting the wordcloud. It is an adapted version of the code found here:
        http://amueller.github.io/word_cloud/auto_examples/simple.html#sphx-glr-auto-examples-simple-py and
        here https://github.com/ddangelov/Top2Vec/blob/master/top2vec/Top2Vec.py

        :param topic_id: id of the topic
        :param n_words: number of words to show in word cloud
        :param background_color: color of the background
        :param width: width of the produced image
        :param height: height of the produced image
        """
        word_score_list = self.get_word_distribution_by_topic_id(topic_id)[:n_words]
        word_score_dict = {tup[0]: tup[1] for tup in word_score_list}
        plt.figure(figsize=(10, 4), dpi=200)
        plt.axis("off")
        plt.imshow(wordcloud.WordCloud(width=width, height=height, background_color=background_color
                                       ).generate_from_frequencies(word_score_dict))
        plt.title("Displaying Topic " + str(topic_id), loc='center', fontsize=24)
        plt.show()

    def get_predicted_topics(self, dataset, n_samples):
        """
        Return the a list containing the predicted topic for each document (length: number of documents).

        :param dataset: CTMDataset to infer topics
        :param n_samples: number of sampling of theta
        :return: the predicted topics
        """
        predicted_topics = []
        thetas = self.get_doc_topic_distribution(dataset, n_samples)

        for idd in range(len(dataset)):
            predicted_topic = np.argmax(thetas[idd] / np.sum(thetas[idd]))
            predicted_topics.append(predicted_topic)
        return predicted_topics

    def get_ldavis_data_format(self, vocab, dataset, n_samples):
        """
        Returns the data that can be used in input to pyldavis to plot
        the topics
        """
        term_frequency = np.ravel(dataset.X_bow.sum(axis=0))
        doc_lengths = np.ravel(dataset.X_bow.sum(axis=1))
        term_topic = self.get_topic_word_distribution()
        doc_topic_distribution = self.get_doc_topic_distribution(dataset, n_samples=n_samples)

        data = {'topic_term_dists': term_topic,
                'doc_topic_dists': doc_topic_distribution,
                'doc_lengths': doc_lengths,
                'vocab': vocab,
                'term_frequency': term_frequency}

        return data

    def get_top_documents_per_topic_id(self, unpreprocessed_corpus, document_topic_distributions, topic_id, k=5):
        probability_list = document_topic_distributions.T[topic_id]
        ind = probability_list.argsort()[-k:][::-1]
        res = []
        for i in ind:
            res.append((unpreprocessed_corpus[i], document_topic_distributions[i][topic_id]))
        return res
    
    #Identify the most relevant image embeddings based on the training_doc_topic_distributions. (Inference over the training dataset)
    def get_top_image_embeddings(self, current_topic_id, training_doc_topic_distributions, image_embeddings, top_n=10):
        #list_top_image_embeddings = []
        #Get the top images with its highest distribution
        current_list_indices = (-training_doc_topic_distributions).argsort(axis=0).T[current_topic_id]
        #list_top_image_embeddings.append(self.train_data.X_image[current_list_indices[:top_n]])
        #return list_top_image_embeddings
        #return torch.Tensor(self.train_data.X_image[current_list_indices[:top_n]])
        
            
        return torch.Tensor(image_embeddings[current_list_indices[:top_n]])
    
    
    #Get list of list of image embeddings. We get the embeddings for all the topics
    #Identify the most relevant image embeddings based on the training_doc_topic_distributions. (Inference over the training dataset)
    def get_list_top_image_embeddings(self, training_doc_topic_distributions,image_embeddings, top_n=10):
        list_top_image_embeddings = []
        for current_topic_id in range(self.n_components):            
            list_top_image_embeddings.append(self.get_top_image_embeddings(current_topic_id, training_doc_topic_distributions,image_embeddings,  top_n))        
        return list_top_image_embeddings

class ZeroShotTM(CTM_Multimodal):
    """ZeroShotTM, as described in https://arxiv.org/pdf/2004.07737v1.pdf

    """

    def __init__(self, **kwargs):
        inference_type = "zeroshot"
        super().__init__(**kwargs, inference_type=inference_type)


class CombinedTM(CTM_Multimodal):
    """CombinedTM, as described in https://arxiv.org/pdf/2004.03974.pdf

    """

    def __init__(self, **kwargs):
        inference_type = "combined"
        super().__init__(**kwargs, inference_type=inference_type)
