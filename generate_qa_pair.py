import json
import pandas as pd
from path_env import FilePathConfig

def generate_ans_type_qa_pair(origin_data_path, qa_pair_path, kg_path):
    # 生成答案类型的qa对
    with open(kg_path) as fp:
        candidates = json.load(fp)["candidates"]
    
    results = []
    df = pd.read_excel(origin_data_path, keep_default_na=False)
    for idx, row in df.iterrows():
        ans_type = row["答案类型"]
        query = row["用户问题"]
        for at in candidates:
            dump = dict()
            dump["id"] = idx
            dump["query"] = query
            dump["ans"] = at
            if at == ans_type:
                dump["label"] = 0
            else:
                dump["label"] = 1
            results.append(dump)

    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    with open(qa_pair_path, "w") as fp:
        fp.write(json_str)

def generate_attr_qa_pair(origin_data_path, qa_pair_path, kg_path):
    # 生成属性的qa对
    with open(kg_path) as fp:
        candidates = json.load(fp)["candidates"]
    
    results = []
    df = pd.read_excel(origin_data_path, keep_default_na=False)
    for idx, row in df.iterrows():
        attr_names = list(row["属性名"].split("|"))
        attr_sub_names = set()
        query = row["用户问题"]
        for name in attr_names:
            sub_name = name.split("-")[-1]
            attr_sub_names.add(sub_name)

        for attr in candidates:
            dump = dict()
            dump["id"] = idx
            dump["query"] = query
            dump["ans"] = attr
            if attr in attr_sub_names:
                dump["label"] = 0
            else:
                dump["label"] = 1
            results.append(dump)

    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    with open(qa_pair_path, "w") as fp:
        fp.write(json_str)

def generate_qa_pair(path_env):
    origin_data_path = path_env.train_set_path
    ans_type_kg_path = path_env.ans_type_kg_path
    attr_kg_path = path_env.attr_kg_path
    ans_type_qa_pair_path = path_env.ans_type_qa_pair_path
    attr_qa_pair_path = path_env.attr_qa_pair_path

    generate_ans_type_qa_pair(origin_data_path, ans_type_qa_pair_path, ans_type_kg_path)
    generate_attr_qa_pair(origin_data_path, attr_qa_pair_path, attr_kg_path)

if __name__ == "__main__":
    path_env = FilePathConfig()
    generate_qa_pair(path_env)