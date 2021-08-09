import pandas as pd
from path_env import  FilePathConfig

def word_count(data_list):
    # 统计单词出现的频次，并将其降序排列，得出出现频次最多的单词
    dic = {}
    for data_str in data_list:
        words = list(data_str)
        for word in words:
            if(word in dic):
                dic[word] += 1
            else:
                dic[word] = 1
    word_count_sorted = sorted(dic.items(), key=lambda item:item[1], reverse=True)
    return  word_count_sorted

def generate_vocab(path_env, vocab_size=5000):
    train_path = path_env.train_set_path
    vocab_path = path_env.vocab_path
    df = pd.read_excel(train_path)
    data_list = []
    for idx, row in df.iterrows():
        data_list.append(row["用户问题"])
        data_list.append(row["答案类型"])
        data_list.append(row["属性名"])
    word_count_sorted = word_count(data_list)
    vocab_size = min(len(word_count_sorted), vocab_size)
    with open(vocab_path, "w") as fp:
        fp.write("<pad>" + "\n")
        fp.write("<unk>" + "\n")
        for i in range(vocab_size):
            fp.write(word_count_sorted[i][0] + "\n")


if __name__ == "__main__":
    path_env = FilePathConfig()
    generate_vocab(path_env)
    