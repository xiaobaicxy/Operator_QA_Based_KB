import json
import pandas as pd
from path_env import FilePathConfig

def generate_ans_type_candidates(data_path, ans_type_kg_path):
    # 根据训练集生成答案类型的候选集
    df = pd.read_excel(data_path, keep_default_na=False)
    ans_types = set()
    for idx, row in df.iterrows():
        ans_types.add(row["答案类型"])
    candidates = sorted(list(ans_types)) # 排序，防止每次运行的结果不一样
    dic = {"candidates": candidates}
    json_str = json.dumps(dic, indent=4, ensure_ascii=False)
    with open(ans_type_kg_path, "w") as fp:
        fp.write(json_str) 

def generate_attr_candidates(data_path, attr_kg_path):
    # 根据训练集生成属性的候选集
    df = pd.read_excel(data_path, keep_default_na=False)
    attrs = set()
    for idx, row in df.iterrows():
        attr_names = list(row["属性名"].replace("｜", "|").replace("||", "|").split("|")) # replace，处理错误数据
        for name in attr_names:
            sub_name = name.split("-")[-1]
            attrs.add(sub_name)
    candidates = sorted(list(attrs))
    dic = {"candidates": candidates}
    json_str = json.dumps(dic, indent=4, ensure_ascii=False)
    with open(attr_kg_path, "w") as fp:
        fp.write(json_str)

def generate_ans_tree_from_train_set(data_path, ans_tree_path):
    # 根据训练集生成用于推理用户问题答案的知识树，后续会根据用户问题的属性+实体+约束属性值来推理出问题的答案
    df = pd.read_excel(data_path, keep_default_na=False)
    results = []
    for idx, row in df.iterrows():
        ans_type = row["答案类型"]
        if ans_type == "并列句":
            continue

        attr_name = row["属性名"]
        f_attr_name = attr_name.split("-")[-1]
        
        entity = row["实体"]

        c_attr_val = str(row["约束属性值"]).replace("｜", "|").replace("||", "|")
        c_attr_vals = []
        if c_attr_val != "":
            c_attr_vals = c_attr_val.split("|")
            c_attr_vals.sort() # 排序，为了后续做相同判断

        answers = list(str(row["答案"]).replace("｜", "|").replace("||", "|").split("|"))
        ans = set(answers)

        not_find = True
        for result in results:
            if result["attr"] == f_attr_name and result["entity"] == entity and result["constraint"] == c_attr_vals:
                result["ans"].union(ans)
                not_find = False
                break
        
        if not_find:
            results.append(
                {
                    "attr": f_attr_name,
                    "entity": entity,
                    "constraint": c_attr_vals,
                    "ans": ans
                }
            )

    for result in results:
        result["ans"] = list(result["ans"])

    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    with open(ans_tree_path, "w") as fp:
        fp.write(json_str)

def generate_ans_tree_from_triples(triples_path, ans_tree_path, attr_kg_path):
    # 根据给定的triples生成拥有推理用户问题答案的知识树
    with open(attr_kg_path) as fp:
        attr_candidates = json.load(fp)["candidates"]
    results = []
    for loop in range(2):
        with open(triples_path, "r") as fp:
            for line in fp.readlines():
                line = line.strip().replace(" _", "_").replace("｜", "|").replace("||", "|").replace(" <", "<").replace(" >", ">").replace("><", "> <")
                
                entity, attr, ans = line.split()
                
                entity = entity[1:-1] # 去掉“<” 和 “>”
                attr = attr[1:-1]
                ans = ans[1:-1]
                if attr == "档位介绍表":
                    continue
                entity_split = list(entity.split("_"))
                group_id = ""
                if len(entity_split) == 1:
                    entity = entity_split[0]
                else:
                    entity = entity_split[-2]
                    group_id = entity_split[-1]

                if attr in attr_candidates and loop == 0:
                    not_find = True
                    for result in results:
                        if result["entity"] == entity and result["attr"] == attr and result["group_id"] == group_id:
                            if attr in attr_candidates:
                                result["ans"].union(set(list(ans.split("|"))))
                            not_find = False
                            break
                    if not_find:
                        results.append({
                            "entity": entity,
                            "attr": attr,
                            "group_id": group_id,
                            "ans": set(list(ans.split("|")))
                        })
                if attr not in attr_candidates and loop == 1: # 属性没在属性备选值中出现过，则将其作为约束属性值
                    for result in results:
                        if result["entity"] == entity and result["group_id"] == group_id:
                            result["constraint"] = set(list(ans.split("|")))
                
    for result in results:
        if "constraint" in result:
            result["constraint"] = list(result["constraint"])
        else:
            result["constraint"] = []
        result["ans"] = list(result["ans"])
    
    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    with open(ans_tree_path, "w") as fp:
        fp.write(json_str)

def generate_ans_tree_for_compare(data_path, ans_tree_path):
    # 为比较句生成单独的kg
    df = pd.read_excel(data_path, keep_default_na=False)
    results = []
    for idx, row in df.iterrows():
        if row["答案类型"] != "比较句":
            continue

        attr_name = row["属性名"]
        f_attr_name = attr_name.split("-")[-1]
        
        entity = row["实体"]

        c_attr_val = str(row["约束属性值"]).replace("｜", "|").replace("||", "|")
        c_attr_vals = []
        if c_attr_val != "":
            c_attr_vals = c_attr_val.split("|")
            c_attr_vals.sort()

        answers = list(str(row["答案"]).replace("｜", "|").replace("||", "|").split("|"))
        ans = set(answers)

        not_find = True
        for result in results:
            if result["attr"] == f_attr_name and result["entity"] == entity and result["constraint"] == c_attr_vals:
                result["ans"].union(ans)
                not_find = False
                break
        
        if not_find:
            results.append(
                {
                    "attr": f_attr_name,
                    "entity": entity,
                    "constraint": c_attr_vals,
                    "ans": ans
                }
            )
    for result in results:
        result["ans"] = list(result["ans"])
    
    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    with open(ans_tree_path, "w") as fp:
        fp.write(json_str)

def generate_kg(path_env):
    train_set_path = path_env.train_set_path
    triples_path = path_env.triples_path
    ans_type_kg_path = path_env.ans_type_kg_path
    attr_kg_path = path_env.attr_kg_path
    ans_tree_from_train_set = path_env.ans_tree_from_train_set
    ans_tree_from_triples = path_env.ans_tree_from_triples
    ans_tree_for_compare = path_env.ans_tree_for_compare

    generate_ans_type_candidates(train_set_path, ans_type_kg_path)
    generate_attr_candidates(train_set_path, attr_kg_path)

    generate_ans_tree_from_train_set(train_set_path, ans_tree_from_train_set)
    generate_ans_tree_from_triples(triples_path, ans_tree_from_triples, attr_kg_path)
    generate_ans_tree_for_compare(train_set_path, ans_tree_for_compare)

if __name__ == "__main__":
    path_env = FilePathConfig()
    generate_kg(path_env)

