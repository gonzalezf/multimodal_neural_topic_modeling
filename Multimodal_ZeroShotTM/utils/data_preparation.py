#There is a hugging face function (tokenize) that I need to look at
import numpy as np
from sentence_transformers import SentenceTransformer
import scipy.sparse
import warnings
from Multimodal_ZeroShotTM.datasets.dataset import CTMDataset, MultimodalCTMDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import torch
import pickle


def get_bag_of_words(data, min_length):
    """
    Creates the bag of words
    """
    vect = [
        np.bincount(x[x != np.array(None)].astype("int"), minlength=min_length)
        for x in data
        if np.sum(x[x != np.array(None)]) != 0
    ]

    vect = scipy.sparse.csr_matrix(vect)
    return vect


def bert_embeddings_from_file(
    text_file, sbert_model_to_load, batch_size=200, max_seq_length=None
):
    """
    Creates SBERT Embeddings from an input file, assumes one document per line
    """

    model = SentenceTransformer(sbert_model_to_load)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    with open(text_file, encoding="utf-8") as filino:
        texts = list(map(lambda x: x, filino.readlines()))

    check_max_local_length(max_seq_length, texts)
    
    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))
'''
def bert_embeddings_from_list(texts, sbert_model_to_load, batch_size=200, max_seq_length=None):
    
    model = SentenceTransformer(sbert_model_to_load)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
        print('Max_seq_length set up to:', max_seq_length)
    else:
        if hasattr(model._first_module(), 'max_seq_length'):
            max_seq_length = model._first_module().max_seq_length
            print('Identifying the max_seq_length from the model: ', max_seq_length)


    texts = check_max_local_length(max_seq_length, texts)
    
    #https://github.com/UKPLab/sentence-transformers/issues/1269
    
    tokenizer = model._first_module().processor.tokenizer

    def truncate_sentence(sentence, tokenizer):
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
            return truncate_sentence(cur_sentence, tokenizer)
        
        else:
            return cur_sentence
    texts = [truncate_sentence(text, tokenizer) for text in texts]
    


    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))
'''

'''
def check_max_local_length(max_seq_length, texts):


    max_local_length = np.max([len(t.split()) for t in texts])
    if max_local_length > max_seq_length:
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(f"the longest document in your collection has {max_local_length} words, the model instead "
                      f"truncates to {max_seq_length} tokens.")
        print(f"the longest document in your collection has {max_local_length} words, the model instead "
                      f"truncates to {max_seq_length} tokens.")
        print('I am truncating the documents to the max_seq_length you specified - NEED TO REVIEW THIS. DOES IT MAKES SENSE??')
        #truncated_texts = [text[:max_seq_length] for text in texts]
        truncated_texts = [' '.join(t.split()[:max_seq_length]) for t in texts] 
        print('This is the new max length in the docs:', np.max([len(t.split()) for t in truncated_texts]), 'and this is the max_seq_length:', max_seq_length)
        return truncated_texts
    else:
        print('Not truncating terms...')
        print('This is the max length in the docs: *number of terms', np.max([len(t.split()) for t in texts]), 'and this is the max_seq_length:', max_seq_length)
        print('This is the max length in the docs *caracters):', np.max([len(t) for t in texts]), 'and this is the max_seq_length:', max_seq_length)

        return texts
'''


def bert_embeddings_from_list(
    texts, sbert_model_to_load, batch_size=200, max_seq_length=None
):
    """
    Creates SBERT Embeddings from a list
    """
    model = SentenceTransformer(sbert_model_to_load)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    check_max_local_length(max_seq_length, texts)

    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


def check_max_local_length(max_seq_length, texts):
    max_local_length = np.max([len(t.split()) for t in texts])
    if max_local_length > max_seq_length:
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            f"the longest document in your collection has {max_local_length} words, the model instead "
            f"truncates to {max_seq_length} tokens."
        )


class TopicModelDataPreparation:
    
    #It can receive either the image embeddings path or the image embeddings itself
    def __init__(
        self, contextualized_model=None, show_warning=True, max_seq_length=None,
        image_embeddings_path = None, image_embeddings_index_path = None, image_embeddings = None, image_embeddings_index = None
    ):
        self.contextualized_model = contextualized_model
        self.vocab = []
        self.id2token = {}
        self.vectorizer = None
        self.label_encoder = None
        self.show_warning = show_warning
        self.max_seq_length = max_seq_length

        if image_embeddings != None:
            self.image_embeddings = image_embeddings
            print('Image embeddings  - Size', len(self.image_embeddings))

        if image_embeddings_index != None:
            self.image_embeddings_index = image_embeddings_index
            print('Image embeddings index  - Size', len(self.image_embeddings_index))
       
        if image_embeddings_path != None:
            self.image_embeddings_path = image_embeddings_path
            self.image_embeddings = torch.load(image_embeddings_path)
            print('Image embeddings loaded - Size', len(self.image_embeddings))
        
        if image_embeddings_index_path != None:
            with open(image_embeddings_index_path, 'rb') as handle:
                self.image_embeddings_index = pickle.load(handle)
            print('Image embeddings index loaded - Size', len(self.image_embeddings_index))
        

    def load(self, contextualized_embeddings, bow_embeddings, id2token, labels=None, image_embeddings = None, image_embeddings_index = None):
        return CTMDataset(
            X_contextual=contextualized_embeddings,
            X_bow=bow_embeddings,
            idx2token=id2token,
            labels=labels,
            image_embeddings = image_embeddings,
            image_embeddings_index = image_embeddings_index 
        )

    def fit(
        self, text_for_contextual, text_for_bow, labels=None, custom_embeddings=None, multimodal=None, commonsense_inferences=None):
        """
        This method fits the vectorizer and gets the embeddings from the contextual model

        :param text_for_contextual: list of unpreprocessed documents to generate the contextualized embeddings
        :param text_for_bow: list of preprocessed documents for creating the bag-of-words
        :param custom_embeddings: np.ndarray type object to use custom embeddings (optional).
        :param labels: list of labels associated with each document (optional).
        """

        if custom_embeddings is not None:
            assert len(text_for_contextual) == len(custom_embeddings)

            if text_for_bow is not None:
                assert len(custom_embeddings) == len(text_for_bow)

            if type(custom_embeddings).__module__ != "numpy":
                raise TypeError(
                    "contextualized_embeddings must be a numpy.ndarray type object"
                )

        if text_for_bow is not None:
            assert len(text_for_contextual) == len(text_for_bow)

        if self.contextualized_model is None and custom_embeddings is None:
            raise Exception(
                "A contextualized model or contextualized embeddings must be defined"
            )

        self.vectorizer = CountVectorizer()

        train_bow_embeddings = self.vectorizer.fit_transform(text_for_bow)

        # if the user is passing custom embeddings we don't need to create the embeddings using the model

        if custom_embeddings is None:
            train_contextualized_embeddings = bert_embeddings_from_list(
                text_for_contextual,
                sbert_model_to_load=self.contextualized_model,
                max_seq_length=self.max_seq_length,
            )
        else:
            train_contextualized_embeddings = custom_embeddings
            
        #Now generate commonsense inference embeddings if needed
        if commonsense_inferences != None:
            commonsense_inferences_embeddings = bert_embeddings_from_list(
                commonsense_inferences,
                sbert_model_to_load=self.contextualized_model,
                max_seq_length=self.max_seq_length,
            )
            train_contextualized_embeddings = np.concatenate((train_contextualized_embeddings, commonsense_inferences_embeddings), axis=1)
            print('shape train_contextualized_embeddings new', train_contextualized_embeddings.shape)
        
        self.vocab = self.vectorizer.get_feature_names_out() #old version package
        #self.vocab = self.vectorizer.get_feature_names()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}

        if labels:
            self.label_encoder = OneHotEncoder()
            encoded_labels = self.label_encoder.fit_transform(
                np.array([labels]).reshape(-1, 1)
            )
        else:
            encoded_labels = None
        if multimodal == None:
            return CTMDataset(
                X_contextual=train_contextualized_embeddings,
                X_bow=train_bow_embeddings,
                idx2token=self.id2token,
                labels=encoded_labels                
            )
        else:            
            return MultimodalCTMDataset(
                X_contextual=train_contextualized_embeddings,
                X_bow=train_bow_embeddings,
                idx2token=self.id2token,
                labels=encoded_labels,
                image_embeddings = self.image_embeddings.to("cpu"), 
                image_embeddings_index = self.image_embeddings_index,
            )

    def transform_multimodal(
        self,
        text_for_contextual=None,
        text_for_bow=None,
        custom_embeddings=None,
        labels=None,
        multimodal = None,
        image_embeddings = None,
        image_embeddings_index = None
    ):
        """
        This method create the input for the prediction. Essentially, it creates the embeddings with the contextualized
        model of choice and with trained vectorizer.

        If text_for_bow is missing, it should be because we are using ZeroShotTM

        :param text_for_contextual: list of unpreprocessed documents to generate the contextualized embeddings
        :param text_for_bow: list of preprocessed documents for creating the bag-of-words
        :param custom_embeddings: np.ndarray type object to use custom embeddings (optional).
        :param labels: list of labels associated with each document (optional).
        """

        if custom_embeddings is not None:
            assert len(text_for_contextual) == len(custom_embeddings)

            if text_for_bow is not None:
                assert len(custom_embeddings) == len(text_for_bow)

        if text_for_bow is not None:
            assert len(text_for_contextual) == len(text_for_bow)

        if self.contextualized_model is None:
            raise Exception(
                "You should define a contextualized model if you want to create the embeddings"
            )

        if text_for_bow is not None:
            test_bow_embeddings = self.vectorizer.transform(text_for_bow)
        
        else:
            # dummy matrix
            if self.show_warning:
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(
                    "The method did not have in input the text_for_bow parameter. This IS EXPECTED if you "
                    "are using ZeroShotTM in a cross-lingual setting"
                )

            # we just need an object that is matrix-like so that pytorch does not complain
            test_bow_embeddings = scipy.sparse.csr_matrix(
                #np.zeros((len(text_for_contextual), 1))
                np.zeros((len(image_embeddings), 1))
            )
        
        if custom_embeddings is None and text_for_contextual is not None:
            test_contextualized_embeddings = bert_embeddings_from_list(
                text_for_contextual,
                sbert_model_to_load=self.contextualized_model,
                max_seq_length=self.max_seq_length,
            )
        else:
            test_contextualized_embeddings = custom_embeddings

        if labels:
            encoded_labels = self.label_encoder.transform(
                np.array([labels]).reshape(-1, 1)
            )
        else:
            encoded_labels = None
        if multimodal == None:
            return CTMDataset(
                X_contextual=test_contextualized_embeddings,
                X_bow=test_bow_embeddings,
                idx2token=self.id2token,
                labels=encoded_labels,
            )
        else:
            if image_embeddings != None:
                #image_embeddings = self.image_embeddings
                image_embeddings = image_embeddings.to("cpu")
                
            if image_embeddings_index == None:
                #image_embeddings_index = self.image_embeddings_index                
                pass 
            return MultimodalCTMDataset(
                X_contextual=test_contextualized_embeddings,
                X_bow=test_bow_embeddings,
                idx2token=self.id2token,
                labels=encoded_labels,
                image_embeddings = image_embeddings, 
                image_embeddings_index = image_embeddings_index,
            ) 


#Original transform - At least before implementing the multimodal one
    def transform(
            self,
            text_for_contextual,
            text_for_bow=None,
            custom_embeddings=None,
            labels=None,
        ):
            """
            This method create the input for the prediction. Essentially, it creates the embeddings with the contextualized
            model of choice and with trained vectorizer.

            If text_for_bow is missing, it should be because we are using ZeroShotTM

            :param text_for_contextual: list of unpreprocessed documents to generate the contextualized embeddings
            :param text_for_bow: list of preprocessed documents for creating the bag-of-words
            :param custom_embeddings: np.ndarray type object to use custom embeddings (optional).
            :param labels: list of labels associated with each document (optional).
            """

            if custom_embeddings is not None:
                assert len(text_for_contextual) == len(custom_embeddings)

                if text_for_bow is not None:
                    assert len(custom_embeddings) == len(text_for_bow)

            if text_for_bow is not None:
                assert len(text_for_contextual) == len(text_for_bow)

            if self.contextualized_model is None:
                raise Exception(
                    "You should define a contextualized model if you want to create the embeddings"
                )

            if text_for_bow is not None:
                test_bow_embeddings = self.vectorizer.transform(text_for_bow)
            else:
                # dummy matrix
                if self.show_warning:
                    warnings.simplefilter("always", DeprecationWarning)
                    warnings.warn(
                        "The method did not have in input the text_for_bow parameter. This IS EXPECTED if you "
                        "are using ZeroShotTM in a cross-lingual setting"
                    )

                # we just need an object that is matrix-like so that pytorch does not complain
                test_bow_embeddings = scipy.sparse.csr_matrix(
                    np.zeros((len(text_for_contextual), 1))
                )

            if custom_embeddings is None:
                test_contextualized_embeddings = bert_embeddings_from_list(
                    text_for_contextual,
                    sbert_model_to_load=self.contextualized_model,
                    max_seq_length=self.max_seq_length,
                )
            else:
                test_contextualized_embeddings = custom_embeddings

            if labels:
                encoded_labels = self.label_encoder.transform(
                    np.array([labels]).reshape(-1, 1)
                )
            else:
                encoded_labels = None
            
            return CTMDataset(
                X_contextual=test_contextualized_embeddings,
                X_bow=test_bow_embeddings,
                idx2token=self.id2token,
                labels=encoded_labels,
            )
