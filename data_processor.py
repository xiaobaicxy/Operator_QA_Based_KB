# -*- coding: utf-8 -*-
import os
import pandas as pd
import json
import numpy as np
import random

class DataProcessor:
    def load_train(self, path):
        with open(path) as fp:
            data_list = json.load(fp)
        return data_list
    
    def load_vocab(self, path):
        word2index = dict()
        with open(path, "r") as fp:
            for idx, line in enumerate(fp.readlines()):
                word = line.strip()
                word2index[word] = idx
        return word2index, idx+1

    def load_train_data(self, train_path, vocab_path, max_len=200):
        word2index, vocab_size = self.load_vocab(vocab_path)
        data_list = self.load_train(train_path)
        contexts = []
        queries = []
        labels = []
        for data in data_list:
            label_idx = data["label"]
            label = [0, 0]
            label[label_idx] = 1
            labels.append(label)

            context = data["query"]
            context_feature = []
            words = list(context) # 文本太短，直接分字
            for word in words:
                if word in word2index:
                    context_feature.append(word2index[word])
                else:
                    context_feature.append(word2index["<unk>"]) 
                if len(context_feature) == max_len:
                    break
            context_feature = context_feature + [word2index["<pad>"]] * (max_len - len(context_feature))
            contexts.append(context_feature)

            query = data["ans"]
            query_feature = []
            words = list(query)
            for word in words:
                if word in word2index:
                    query_feature.append(word2index[word])
                else:
                    query_feature.append(word2index["<unk>"])
                if len(query_feature) == max_len:
                    break
            query_feature = query_feature + [word2index["<pad>"]] * (max_len - len(query_feature))
            queries.append(query_feature)
            
        return contexts, queries, labels, vocab_size
    
    
    def create_train_set(self, train_path, vocab_path, batch_size=64, dev_scale=0.):
        def get_batchs(num, batch_size):
            idxs = [i for i in range(num)]
            batch_first_idx = np.arange(start=0, stop=num, step=batch_size)
            np.random.shuffle(idxs)
            batches = []
            for first_idx in batch_first_idx:
                batch = [idxs[i] for i in range(first_idx, min(first_idx+batch_size, num))]
                batches.append(batch)
            return batches
        if not os.path.exists(vocab_path):
            self.create_vocab(train_path, vocab_path)
        contexts, queries, labels, vocab_size = self.load_train_data(train_path, vocab_path)
        
        batches = get_batchs(len(contexts), batch_size)
        train_set = []
        dev_set = []
        for batch in batches:
            batch_context = []
            batch_query = []
            batch_label = []
            for idx in batch:
                batch_context.append(contexts[idx])
                batch_query.append(queries[idx])
                batch_label.append(labels[idx])
            if (random.random() > dev_scale):
                train_set.append((batch_context, batch_query, batch_label))
            else:
                dev_set.append((batch_context, batch_query, batch_label))
        return train_set, dev_set, vocab_size

    def load_infer_data(self, infer_path, kg_path, vocab_path, max_len=200):
        word2index, vocab_size = self.load_vocab(vocab_path)
        df = pd.read_excel(infer_path, keep_default_na=False)
        contexts = []
        queries = []
        ids = []
        c_ids = []
        with open(kg_path) as fp:
            candidates = json.load(fp)["candidates"]

        for idx, row in df.iterrows():
            q_id = row["id"]
            context = row["query"]
            for c_idx, query in enumerate(candidates):
                c_ids.append(c_idx)
                ids.append(q_id)
                context_feature = []
                words = list(context)
                for word in words:
                    if word in word2index:
                        context_feature.append(word2index[word])
                    else:
                        context_feature.append(word2index["<unk>"]) 
                    if len(context_feature) == max_len:
                        break
                context_feature = context_feature + [word2index["<pad>"]] * (max_len - len(context_feature))
                contexts.append(context_feature)

                query_feature = []
                words = list(query)
                for word in words:
                    if word in word2index:
                        query_feature.append(word2index[word])
                    else:
                        query_feature.append(word2index["<unk>"])
                    if len(query_feature) == max_len:
                        break
                query_feature = query_feature + [word2index["<pad>"]] * (max_len - len(query_feature))
                queries.append(query_feature)
        return contexts, queries, ids, c_ids, vocab_size
    

    def create_infer_set(self, infer_path, kg_path, vocab_path, batch_size=64):
        def get_batchs(num, batch_size):
            idxs = [i for i in range(num)]
            batch_first_idx = np.arange(start=0, stop=num, step=batch_size)
            batches = []
            for first_idx in batch_first_idx:
                batch = [idxs[i] for i in range(first_idx, min(first_idx+batch_size, num))]
                batches.append(batch)
            return batches

        contexts, queries, ids, q_ids, vocab_size = self.load_infer_data(infer_path, kg_path, vocab_path)
        
        batches = get_batchs(len(contexts), batch_size)
        infer_set = []
        for batch in batches:
            batch_context = []
            batch_query = []
            batch_ids = []
            batch_q_ids = []
            for idx in batch:
                batch_context.append(contexts[idx])
                batch_query.append(queries[idx])
                batch_ids.append(ids[idx])
                batch_q_ids.append(q_ids[idx])
            infer_set.append((batch_context, batch_query, batch_ids, batch_q_ids))

        return infer_set, vocab_size
