#Create a class to load datasets given a csv file path
import pandas as pd
import json
import torch
import pickle

class ImportDatasetTopicModeling:
    def __init__(self, 
                 name_dataset, 
                 datasets_path,
                 cluster='local', 
                 size_sampling=None, 
                 column_text_name = 'com',
                 image_id =  'photo_flickr_id'
                 ) -> None:
        self.name_dataset = name_dataset
        self.cluster = cluster
        self.size_sampling = size_sampling
        self.datasets_path = datasets_path
        self.column_text_name = column_text_name
        self.image_id = image_id
    
    def get_dataset(self):
        
        #read the json file with the paths
        with open(self.datasets_path) as json_file:
            datasets_path = json.load(json_file)
        self.base_path_dir  = datasets_path['base_path_dir'][self.cluster]
        self.base_path_dir_embeddings = datasets_path['base_path_dir_embeddings'][self.cluster]
    
        self.word2vec_path = self.base_path_dir+'GoogleNews-vectors-negative300.bin.gz'
        self.path_file = datasets_path[self.name_dataset][self.cluster]['path_file']
        self.image_emb_file = datasets_path[self.name_dataset][self.cluster]['image_emb_file']
        self.image_emb_file_csv = datasets_path[self.name_dataset][self.cluster]['image_emb_file_csv']
        self.image_embeddings_index_path = datasets_path[self.name_dataset][self.cluster]['image_embeddings_index_path']

        df = pd.read_csv(self.path_file)
        if self.size_sampling!=None:
            df = df.sample(self.size_sampling)
            df.to_csv(self.path_file+'_size_sampling_'+str(self.size_sampling)+'.csv', index=False)
        
        documents = df[self.column_text_name].tolist()
       
        print('Rows in dataset:', len(df))
        print('Number of documents: ', len(documents))
        return documents, df
    
    def get_word2vec_path(self):
        return self.word2vec_path
    
    def get_sbert_model_path(self, model_name):
        if  model_name == 'clip':
            return self.base_path_dir_embeddings+"clip-ViT-L-14"
        if  model_name == 'clip_32':
            return self.base_path_dir_embeddings+"clip-ViT-B-32"
        if model_name == 'sbert':
            return self.base_path_dir_embeddings+"all-mpnet-base-v2"
        
    
    def get_image_urls(self, df):
        return list(df[self.image_id])

    def get_image_emb_file(self):
        return self.image_emb_file
    
    def get_image_emb_file_csv(self):
        return self.image_emb_file_csv
    
    def get_image_embeddings(self):
        return torch.load(self.image_emb_file)
    
    def get_image_embeddings_index(self):        
        with open(self.image_embeddings_index_path, 'rb') as handle:
            image_embeddings_index = pickle.load(handle)
        return image_embeddings_index
    

    def convert_list_sent_to_str(self, list_sentences):
        return ', '.join(list_sentences)