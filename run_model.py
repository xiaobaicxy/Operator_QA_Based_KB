# -*- coding: utf-8 -*-
import torch
import copy
import torch.nn as nn
import json
import argparse

from config import Config
from data_processor import DataProcessor
from bidaf import BidafModel
from path_env import FilePathConfig

def test(model, dev_set, loss_func, device):
    model.eval()
    loss_val = 0.
    corrects = 0.
    data_size = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for batch_data in train_set:
        context, query, label = batch_data
        
        context = torch.LongTensor(context).to(device)
        query = torch.LongTensor(query).to(device)
        label = torch.FloatTensor(label).to(device)

        c_len = (context != 0).sum(dim=-1).long()
        q_len = (query != 0).sum(dim=-1).long()

        preds = model((context, c_len), (query, q_len))
        loss = loss_func(preds, label)

        data_size += context.size(0)
        loss_val += loss.item() * context.size(0)

        label = torch.argmax(label, dim=1)
        preds = torch.argmax(preds, dim=1)
        corrects += torch.sum(preds == label).item()
        # 因为正类标签为0,负类为1,所以tp,tn等计算方法需要调整
        tn += (label * preds).sum().item()
        tp += ((1 - label) * (1 - preds)).sum().item()
        fn += ((1 - label) * preds).sum().item()
        fp += (label * (1 - preds)).sum().item()

    epsilon = 1e-10
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    loss_val = loss_val / data_size
    acc = corrects / data_size
    print("Dev Loss: {}, Dev Acc: {}, Dev P: {}, Dev R: {}, Dev F1: {}".format(loss_val, acc, precision, recall, f1))
    return f1


def train(model, train_set, dev_set, optimizer, loss_func, epochs, device):
    best_dev_f1 = 0.
    best_train_f1 = 0.
    best_model_params = copy.deepcopy(model.state_dict())
    for ep in range(epochs):
        loss_val = 0.
        corrects = 0.
        data_size = 0

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        model.train()
        for batch_data in train_set:
            context, query, label = batch_data
            
            context = torch.LongTensor(context).to(device)
            query = torch.LongTensor(query).to(device)
            label = torch.FloatTensor(label).to(device)

            c_len = (context != 0).sum(dim=-1).long()
            q_len = (query != 0).sum(dim=-1).long()

            preds = model((context, c_len), (query, q_len))
            loss = loss_func(preds, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            data_size += context.size(0)
            loss_val += loss.item() * context.size(0)
            
            #获取预测的最大概率出现的位置
            preds = torch.argmax(preds, dim=1)
            label = torch.argmax(label, dim=1)
            corrects += torch.sum(preds == label).item()

            # 因为正类标签为0,负类为1,所以tp,tn等计算方法需要调整
            tn += (label * preds).sum().item()
            tp += ((1 - label) * (1 - preds)).sum().item()
            fn += ((1 - label) * preds).sum().item()
            fp += (label * (1 - preds)).sum().item()

        epsilon = 1e-10
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2* (precision*recall) / (precision + recall + epsilon)
            
        loss_val = loss_val / data_size
        acc = corrects / data_size
        
        print("################# Train ###################")
        print("Loss: {}, Acc: {}, P: {}, R: {}, F1: {}".format(loss_val, acc, precision, recall, f1))
        
        if len(dev_set) > 0:
            eval_f1 = test(model, dev_set, loss_func, device)
            if eval_f1 > best_dev_f1:
                best_dev_f1 = eval_f1
                best_model_params = copy.deepcopy(model.state_dict())
        else:
            if f1 > best_train_f1 and best_train_f1 < 0.999: # 防止过拟合，best_train_f1>=99.9%时不再继续更新
                best_train_f1 = f1
                best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    return model


def infer(model, infer_set, result_path, device):
    model.eval()
    results = dict()
    for batch_data in infer_set:
        context, query, ids, q_ids = batch_data

        context = torch.LongTensor(context).to(device)
        query = torch.LongTensor(query).to(device)

        c_len = (context != 0).sum(dim=-1).long()
        q_len = (query != 0).sum(dim=-1).long()

        probs = model((context, c_len), (query, q_len))
        confidences = probs[:,0].tolist()
        for i in range(len(ids)):
            if ids[i] not in results:
                results[ids[i]] = []
            results[ids[i]].append({"ans_id": q_ids[i], "confidence": confidences[i]})
            
    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    with open(result_path, "w") as fp:
        fp.write(json_str)

def run_model(path_env, config, run_type, data_type):
    processor = DataProcessor()

    if data_type == "ans_type":
        train_path = path_env.ans_type_qa_pair_path
        model_path = path_env.ans_type_model
        kg_path = path_env.ans_type_kg_path
        result_path = path_env.ans_type_infer_result
    else:
        train_path = path_env.attr_qa_pair_path
        model_path = path_env.attr_model
        kg_path = path_env.attr_kg_path
        result_path = path_env.attr_infer_result

    if run_type == "train":
        train_set, dev_set, vocab_size = processor.create_train_set(train_path,
                                                                    path_env.vocab_path,
                                                                    batch_size=config.batch_size,
                                                                    dev_scale=config.dev_scale)
        config.vocab_size = vocab_size

        model = BidafModel(config).to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_func = nn.BCELoss()

        model = train(model, train_set, dev_set, optimizer, loss_func, config.epochs, config.device)
        torch.save(model.state_dict(), model_path)

    else:
        infer_set, vocab_size = processor.create_infer_set(path_env.infer_set_path,
                                                           kg_path,
                                                           path_env.vocab_path,
                                                           batch_size=config.batch_size)
        config.vocab_size = vocab_size
        model = BidafModel(config).to(config.device)
        model.load_state_dict(torch.load(model_path))
        infer(model, infer_set, result_path, config.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rt", "--run_type", type=str, choices=["train", "infer"], default="train", help="run type: train or infer, default: train")
    parser.add_argument("-dt", "--data_type", type=str, choices=["ans_type", "attr"], default="ans_type", help="data type: ans_type or attr, default: ans_type")
    args = parser.parse_args()
    run_type = args.run_type
    data_type = args.data_type
    path_env = FilePathConfig()
    config = Config()
    run_model(path_env, config, run_type, data_type)
