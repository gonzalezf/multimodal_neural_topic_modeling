from contextualized_topic_models.evaluation.measures import CoherenceNPMI, InvertedRBO, CoherenceCV, TopicDiversity
from matplotlib import pyplot as plt

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
from scipy.spatial.distance import cosine
import abc

from contextualized_topic_models.evaluation.rbo import rbo
import numpy as np
import itertools

import pyLDAvis as vis
import warnings

from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity
import itertools
import torch
from itertools import combinations
from sentence_transformers import SentenceTransformer



class Measure:
    def __init__(self):
        pass

    def score(self):
        pass

class DiversityBetaImageEmbeddings(Measure): #This is for  ctm multimodal
    def __init__(self, best_components_img):
        super().__init__()
        self.img_embedding_beta = best_components_img
        
         

    def score(self):
        return CoherenceImageEmbeddings([self.img_embedding_beta]).score(topk=len(self.img_embedding_beta))
        
        
'''
Based on 


https://link.springer.com/chapter/10.1007/978-3-030-80599-9_4
https://github.com/silviatti/topic-model-diversity/blob/master/diversity_metrics.py
https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/similarity_metrics.py
pairwise_word_embedding_distance(topics, word_embedding_model, topk=10):
'''


class Measure:
    def __init__(self):
        pass

    def score(self):
        pass

class ImageEPS(Measure):
    def __init__(self, list_top_image_embeddings = None):
        super().__init__()
        #if list_top_image_embeddings is None:
            
            #training_doc_topic_distributions = self.get_doc_topic_distribution(training_dataset, lang_index=3)
            #list_top_image_embeddings = self.get_list_top_image_embeddings(training_doc_topic_distributions)

        self.list_top_image_embeddings = list_top_image_embeddings
    
    def score(self, topk=10):
        if topk > len(self.list_top_image_embeddings[0]):
            raise Exception('Image embeddings in topics are less than topk')
        else:
            count = 0
            sum_sim = 0
            for list1, list2 in combinations(self.list_top_image_embeddings, 2):
                image_counts = 0
                sim = 0
                for image1 in list1[:topk]:
                    for image2 in list2[:topk]:
                        #cos_score = torch.nn.functional.cosine_similarity(topic[i].unsqueeze(0), topic[j].unsqueeze(0), dim=1).item()
                        cos_score = torch.nn.functional.cosine_similarity(image1.unsqueeze(0), image2.unsqueeze(0), dim=1).item()
                        sim = sim + cos_score
                        image_counts = image_counts + 1
                sim = sim / image_counts
                sum_sim = sum_sim + sim
                count = count + 1
            return sum_sim / count
class CoherenceImageEmbeddings(Measure):
    def __init__(self, list_top_image_embeddings):
        """
        :param topics: a list of lists of the top-n most likely image embeddings
        :param word2vec_path: if word2vec_file is specified, it retrieves the
         word embeddings file (in word2vec format) to compute similarities
         between words, otherwise 'word2vec-google-news-300' is downloaded
        :param binary: if the word2vec file is binary
        """
        super().__init__()
        self.top_image_embeddings = list_top_image_embeddings #list of list of top image embeddings. 
        #Size K x N, where K is the number of topics and N is the number of images in each topic
        #IMPORTANT: Make sure that list_top_image_embeddings is a list of lists.We need the top n images from all the topics. This function return the average

    def score(self, topk=10):
        """
        :return: topic coherence computed on the word embeddings similarities
        """

        if topk > len(self.top_image_embeddings[0]):
            raise Exception('Image embeddings in topics are less than topk')
        else:
            arrays = []
            for index, topic in enumerate(self.top_image_embeddings):
                if len(topic) > 0:
                    
                    local_simi = []
                    #for i in range(len(topic)):
                    for i in range(topk):
                        for j in range(i+1, topk): #for j in range(i+1, len(topic)):
                            #local_simi.append(cosine_similarity(topic[i].reshape(1,-1), topic[j].reshape(1,-1)))
                            #local_simi.append(cosine_similarity(topic[i], topic[j]))
                            current_score = torch.nn.functional.cosine_similarity(topic[i].unsqueeze(0), topic[j].unsqueeze(0), dim=1).item()
                            local_simi.append(current_score)
                arrays.append(np.mean(local_simi))
                
            return np.mean(arrays)
        
    



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    #ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
def bert_embeddings_from_list(texts, sbert_model_to_load, batch_size=200, max_seq_length=None):
    """
    Creates SBERT Embeddings from a list
    """
    model = SentenceTransformer(sbert_model_to_load)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
        check_max_local_length(max_seq_length, texts) #This is important I fixex it! THE MAX_SEQ_LENGHT SOMETIMES IS NONE
    #check_max_local_length(max_seq_length, texts #THIS IS WRONG! IT SHOULD NOT BE GHERE
    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


def check_max_local_length(max_seq_length, texts):
    max_local_length = np.max([len(t.split()) for t in texts])
    print('max_seq_length', max_seq_length)
    print('second max_local_length', max_local_length)
    if max_local_length > max_seq_length:
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(f"the longest document in your collection has {max_local_length} words, the model instead "
                      f"truncates to {max_seq_length} tokens.")
        
#I needed to make modifications int he topic evlauation of ctm
#https://github.com/MilaNLProc/contextualized-topic-models/blob/master/contextualized_topic_models/evaluation/measures.py


class Measure:
    def __init__(self):
        pass

    def score(self):
        pass

    
class CoherenceWordEmbeddings(Measure):
    def __init__(self, topics, word2vec_path=None, binary=False, w2v_embeddings = None):
        """
        :param topics: a list of lists of the top-n most likely words
        :param word2vec_path: if word2vec_file is specified, it retrieves the
         word embeddings file (in word2vec format) to compute similarities
         between words, otherwise 'word2vec-google-news-300' is downloaded
        :param binary: if the word2vec file is binary
        """
        super().__init__()
        self.topics = topics
        self.binary = binary
        
        if w2v_embeddings != None:
            self.wv = w2v_embeddings
        else:
            if word2vec_path is None:
                self.wv = api.load('word2vec-google-news-300')
            else:
                self.wv = KeyedVectors.load_word2vec_format(
                    word2vec_path, binary=binary)

    def score(self, topk=10):
        """
        :param topk: how many most likely words to consider in the evaluation
        :return: topic coherence computed on the word embeddings similarities
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            arrays = []
            for index, topic in enumerate(self.topics):
                if len(topic) > 0:
                    local_simi = []
                    for word1, word2 in itertools.combinations(
                            topic[:topk], 2):
                        if (word1 in self.wv.index_to_key
                                and word2 in self.wv.index_to_key):
                            local_simi.append(self.wv.similarity(word1, word2))
                    arrays.append(np.mean(local_simi))
            return np.mean(arrays)
        
def truncate_sentence(sentence, tokenizer,max_seq_length):
    """
    Truncate a sentence to fit the CLIP max token limit (77 tokens including the
    starting and ending tokens).
    
    Args:
        sentence(string): The sentence to truncate.
        tokenizer(CLIPTokenizer): Rretrained CLIP tokenizer.
    """
    
    cur_sentence = sentence
    tokens = tokenizer.encode(cur_sentence)
    
    if len(tokens) > max_seq_length:
        # Skip the starting token, only include 75 tokens
        truncated_tokens = tokens[1:max_seq_length-1]
        cur_sentence = tokenizer.decode(truncated_tokens)
        
        # Recursive call here, because the encode(decode()) can have different result
        return truncate_sentence(cur_sentence, tokenizer,max_seq_length)
    
    else:
        return cur_sentence
def adjust_length_sentences(sbert_model_to_load,unpreprocessed_corpus, max_seq_length):
    model = SentenceTransformer(sbert_model_to_load)
    print(sbert_model_to_load)
    if hasattr(model._first_module(), 'processor'): #sbert
        if hasattr(model._first_module().processor, 'tokenizer'): #sbert
            tokenizer = model._first_module().processor.tokenizer                    
            print('Finding tokenizer - clip')
    elif hasattr(model._first_module(), 'tokenizer'): #clip
      print('Finding tokenizer - sbert')
      tokenizer = model._first_module().tokenizer
    else:
        raise ValueError("Tokenizer not found in the model.")
    print('Identifying the max_seq_length from the model: ', max_seq_length)
    print('Number of sentences received', len(unpreprocessed_corpus))
    unpreprocessed_corpus = [truncate_sentence(sent,tokenizer,max_seq_length) for sent in unpreprocessed_corpus]
    print('Number of final sentences received', len(unpreprocessed_corpus))

    #https://github.com/UKPLab/sentence-transformers/issues/1269

    return unpreprocessed_corpus


        
def compute_scores(ctm, topics, texts, apply_weco=True, apply_rbo=True, apply_npmi=True, apply_cv= True, apply_topic_diversity = True , word2vec_path=None, binary_w2v=True, w2v_embeddings= None, n_terms_topic_diversity=10, top_image_embeddings_training_data=None,top_image_embeddings_beta_img = None, apply_IECO=True,apply_ImageEPS = True , apply_diversity_beta_img_embeddings=True):
    start_time = datetime.now()
    #weco = CoherenceWordEmbeddings(topics)
    #texts = ctm.texts
    if w2v_embeddings == None:
        if word2vec_path == None:
            word2vec_path = ctm.word2vec_path
    dict_metrics = {}
    if apply_weco:
        weco = CoherenceWordEmbeddings(topics, word2vec_path = word2vec_path, binary = binary_w2v, w2v_embeddings = w2v_embeddings)
        dict_metrics['WECO']= weco.score()
    if apply_IECO and top_image_embeddings_training_data is not None:
        #top_image_embeddings should be a list of list of embeddings
        ieco = CoherenceImageEmbeddings(top_image_embeddings_training_data) 
        dict_metrics['IECO_training_data']= ieco.score()#top    
    if apply_IECO and top_image_embeddings_beta_img is not None:
        ieco = CoherenceImageEmbeddings(top_image_embeddings_beta_img) 
        dict_metrics['IECO_beta_img']= ieco.score()#top    

    if apply_ImageEPS and top_image_embeddings_training_data is not None:
        #top_image_embeddings should be a list of list of embeddings
        image_eps = ImageEPS(top_image_embeddings_training_data) 
        dict_metrics['ImageEPS_training_data']= image_eps.score()
    if apply_ImageEPS and top_image_embeddings_beta_img is not None:
        #top_image_embeddings should be a list of list of embeddings
        image_eps = ImageEPS(top_image_embeddings_beta_img) 
        dict_metrics['ImageEPS_beta_img']= image_eps.score()
    if apply_diversity_beta_img_embeddings and hasattr(ctm, 'best_components_img'):
        diversity_beta = DiversityBetaImageEmbeddings(ctm.best_components_img)        
        dict_metrics['diversity_beta_img_embeddings'] = diversity_beta.score()
    
    if apply_rbo:
        rbo = InvertedRBO(topics)
        dict_metrics['RBO'] = rbo.score()
    if apply_npmi:
        npmi = CoherenceNPMI(texts=texts, topics=topics)
        dict_metrics['NPMI'] = npmi.score()
    if apply_cv:
        cv = CoherenceCV(texts=texts, topics=topics)
        dict_metrics['cv'] = cv.score()
    if apply_topic_diversity:
        print('Number of terms for topic diversity: ', n_terms_topic_diversity)
        if len(topics[0]) != n_terms_topic_diversity:
            print('Warning - You are calculating topic diversity over ', n_terms_topic_diversity, ' terms, but the topics have ', len(topics[0]), ' terms. This may cause problems.')
            topic_diversity = TopicDiversity(ctm.get_topic_lists(n_terms_topic_diversity))
        else:
            topic_diversity = TopicDiversity(topics)
        dict_metrics['topic_diversity'] = topic_diversity.score(topk=n_terms_topic_diversity)
    
    
    #return npmi.score(), rbo.score(), weco.score()
    #return (("NPMI:", npmi.score()), ('RBO:' ,rbo.score()),  ('WECO:' ,weco.score()))
    #return (("NPMI:", npmi.score()), ('RBO:' ,rbo.score()))        
    end_time = datetime.now()        
    print('Duration Compute Scores: {}'.format(end_time - start_time))
    return dict_metrics

    #Asumming there are multiple languages


def compute_scores_M3L(ctm, topics, texts, language_idx = 0, apply_weco=True, apply_rbo=True, apply_npmi=True, apply_cv= True, apply_topic_diversity = True , word2vec_path=None, binary_w2v=True, w2v_embeddings= None,n_terms_topic_diversity=10, top_image_embeddings=None, apply_IECO=True, apply_ImageEPS = True):
    start_time = datetime.now()
    #weco = CoherenceWordEmbeddings(topics)
    #texts = ctm.texts
    if w2v_embeddings == None:
        if word2vec_path == None:
            word2vec_path = ctm.word2vec_path
    dict_metrics = {}
    if apply_weco:
        weco = CoherenceWordEmbeddings(topics, word2vec_path = word2vec_path, binary = binary_w2v, w2v_embeddings = w2v_embeddings)
        dict_metrics['WECO']= weco.score()
    if apply_IECO and top_image_embeddings is not None:
        #top_image_embeddings should be a list of list of embeddings
        ieco = CoherenceImageEmbeddings(top_image_embeddings) 
        dict_metrics['IECO']= ieco.score()  
    if apply_ImageEPS and top_image_embeddings is not None:
        #top_image_embeddings should be a list of list of embeddings
        image_eps = ImageEPS(top_image_embeddings) 
        dict_metrics['ImageEPS']= image_eps.score()

    if apply_rbo:
        rbo = InvertedRBO(topics)
        dict_metrics['RBO'] = rbo.score()
    if apply_npmi:
        
        
        npmi = CoherenceNPMI(texts=texts, topics=topics)
        dict_metrics['NPMI'] = npmi.score()
    if apply_cv:
        cv = CoherenceCV(texts=texts, topics=topics)
        dict_metrics['cv'] = cv.score()
    if apply_topic_diversity:
        print('Number of terms for topic diversity: ', n_terms_topic_diversity)
        topic_diversity = TopicDiversity(ctm.get_topic_lists(n_terms_topic_diversity)[language_idx])
        dict_metrics['topic_diversity'] = topic_diversity.score(topk=n_terms_topic_diversity)
    
    
    #return npmi.score(), rbo.score(), weco.score()
    #return (("NPMI:", npmi.score()), ('RBO:' ,rbo.score()),  ('WECO:' ,weco.score()))
    #return (("NPMI:", npmi.score()), ('RBO:' ,rbo.score()))        
    end_time = datetime.now()        
    print('Duration Compute Scores: {}'.format(end_time - start_time))
    return dict_metrics

from random import shuffle
import numpy as np

def plot_cdf(list_counts, xlabel, path, leg=False, islogx=True, xlimit=False, new_xticks=False):
    t_col = "#235dba"
    g_col = "#005916"
    c_col = "#a50808"
    r_col = "#ff9900"
    black = "#000000"
    pink = "#f442f1"
    t_ls = '-'
    r_ls = '--'
    c_ls = ':'
    g_ls = '-.'

    markers = [".", "o", "v", "^", "<", ">", "1", "2"]
    colors = [t_col, c_col, g_col, r_col, black, 'c', 'm', pink]
    line_styles = [t_ls, r_ls, c_ls, g_ls,t_ls, r_ls, c_ls, g_ls, t_ls]
    colors = colors[1:]
    line_styles= line_styles[1:]
    while(len(list_counts) > len(colors)):
        colors = colors + shuffle(colors)
        line_styles = line_styles + shuffle(line_styles)
        
    if xlimit:
        l2 = []
        for l in list_counts:
            l2_1 = [x for x in l if x<=xlimit]
            l2.append(l2_1)
        list_counts = l2
    
    for l in list_counts:
        l.sort()
    fig, ax = plt.subplots(figsize=(6,4))
    yvals = []
    for l in list_counts:
        yvals.append(np.arange(len(l))/float(len(l)-1))
    for i in range(len(list_counts)):
        ax.plot(list_counts[i], yvals[i], color=colors[i], linestyle=line_styles[i])
    if islogx:
        ax.set_xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.grid()
    
    if new_xticks !=False:    
        ax.set_xticks(new_xticks)

    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(13)
    
    if leg:
        plt.legend(leg, loc='best', fontsize=13)
    

    plt.show()
    fig.savefig(path, bbox_inches='tight')

# Special filtering for news20groups    
import string
import re


def filter_docs(doc):    
    doc = str(doc).lower()
    doc =  re.sub(r'\S*@\S*\s?', '', doc) # remove emails
    # remove newline chars using a regex expression
    doc = re.sub(r'\s+', ' ', doc)  
    #remove single quotes using regex
    doc = re.sub(r"\'", "", doc)  
    return doc

def get_ldavis(ctm, n_samples=5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #lda_vis_data = ctm.get_ldavis_data_format(vocab, training_dataset, n_samples =5)
        lda_vis_data = ctm.get_ldavis_data_format(ctm.vocab, ctm.training_dataset, n_samples =n_samples)
        ctm_vis = vis.prepare(**lda_vis_data)
        return ctm_vis