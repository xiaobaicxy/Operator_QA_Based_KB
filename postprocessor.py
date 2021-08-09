import json
import pandas as pd
import operator
import jieba

from path_env import FilePathConfig

def load_synonyms(data_path):
    # 实体的同义词
    results = []
    with open(data_path, "r") as fp:
        for line in fp.readlines():
            line = line.strip()
            base, others = line.split()
            if others == "无":
                results.append([base])
            else:
                result = [base] + list(others.replace("｜", "|").replace("||", "|").split("|"))
                results.append(list(set(result)))
    return results

def is_en_word(s):
    for c in s.lower():
        if not ord(c) in range(97, 123):
            return False
    return True

def match(p, s):
    count = 0
    words = list(jieba.cut(p, cut_all=False))
    total_words = list(jieba.cut(s, cut_all=False))
    for word in words:
        if word in total_words:
            count += 1
        elif is_en_word(word): # 字母单独匹配
            for w in total_words:
                if is_en_word(w):
                    temp_count = 0
                    for char in list(word.lower()):
                        if char in w.lower():
                            temp_count += 1
                    if len(list(word)) <= 3:
                        if temp_count == len(word):
                            count += 1
                            break
                    elif len(list(word)) <= 5:
                        if temp_count >= len(word) - 1:
                            count += 1
                            break
                    else:
                        if temp_count >= len(word) - 2:
                            count += 1
                            break
    if len(words) < 3:
        if count == len(words):
            return True
    elif len(words) < 5:
        if count >= len(words) - 1:
            return True
    else:
        if count >= len(words) - 2:
            return True
    return False

def postprocess_ans_type_result(pre_result_path, post_result_path, kg_path):
    with open(pre_result_path) as fp:
        data = json.load(fp)
    with open(kg_path) as fp:
        candidates = json.load(fp)["candidates"]

    results = dict()
    for data_id in data.keys():
        ans_list = data[data_id]
        ans_list = sorted(ans_list, key=operator.itemgetter("confidence"), reverse=True)
        ans_id = ans_list[0]["ans_id"]
        if data_id not in results:
            results[data_id] = {"ans_type": candidates[ans_id]}

    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    with open(post_result_path, "w") as fp:
        fp.write(json_str)


def postprocess_attr_result(pre_result_path, post_result_path, infer_set_path, post_ans_type_path, kg_path):
    with open(pre_result_path) as fp:
        data = json.load(fp)
    with open(kg_path) as fp:
        candidates = json.load(fp)["candidates"]
    df = pd.read_excel(infer_set_path, keep_default_na=False)
    with open(post_ans_type_path) as fp:
        ans_type_results = json.load(fp)

    query_dict = dict()
    for idx, row in df.iterrows():
        data_id = row["id"]
        query = row["query"]
        query_dict[data_id] = query

    results = dict()
    for data_id in query_dict.keys():
        ans_list = data[str(data_id)]
        ans_list = sorted(ans_list, key=operator.itemgetter("confidence"), reverse=True)
        ans_type = ans_type_results[str(data_id)]["ans_type"]
        if ans_type == "并列句":
            ans_list = ans_list[:2]
        else:
            ans_list = ans_list[:1]

        attrs = []
        for ans in ans_list:
            ans_id = ans["ans_id"]
            attrs.append(candidates[ans_id])
        attr = "|".join(attrs)

        results[data_id] = {
                                "query": query_dict[data_id],
                                "ans_type": ans_type, 
                                "attr": attr
                            }

    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    with open(post_result_path, "w") as fp:
        fp.write(json_str)

def extract_entity(pre_result_path, post_result_path, entity_synonyms_path):
    entity_sysnoyms = load_synonyms(entity_synonyms_path)
    with open(pre_result_path) as fp:
        attr_results = json.load(fp)
    for loop in range(2):
        for data_id in attr_results.keys():
            query = attr_results[data_id]["query"]
            for entities in entity_sysnoyms:
                for entity in entities:
                    if entity == "":
                        continue
                    if loop == 0 and entity.lower() in query.lower(): # 防止大小写不统一，如：GB和gb
                        if "entity" not in attr_results[data_id]:
                            attr_results[data_id]["entity"] = entity
                        else:
                            if len(entity) > len(attr_results[data_id]["entity"]): # 最长匹配
                                attr_results[data_id]["entity"] = entity
                    elif loop == 1: # 测试集中用拼写错误，如plus被拼写成了puls（id=14）、pluas（id=61）等, 使用新的匹配规则
                        if match(entity, query):
                            if "entity" not in attr_results[data_id]:
                                attr_results[data_id]["entity"] = entity
                            else:
                                if len(entity) > len(attr_results[data_id]["entity"]):
                                    attr_results[data_id]["entity"] = entity
    
    for data_id in attr_results.keys():
        if "entity" not in attr_results[data_id]: # 无实体,补空值
            attr_results[data_id]["entity"] = ""
            
    json_str = json.dumps(attr_results, indent=4, ensure_ascii=False)
    with open(post_result_path, "w") as fp:
        fp.write(json_str)

def generate_ans_from_train_kg(pre_result_path, post_result_path, entity_synonyms_path, kg_path):
    # 用训练数据构造的kg推理query的答案
    entity_sysnoyms = load_synonyms(entity_synonyms_path)
    with open(pre_result_path) as fp:
        entity_results = json.load(fp)
    with open(kg_path) as fp:
        kg_trees = json.load(fp)

    results = []
    for data_id in entity_results.keys():
        query = entity_results[data_id]["query"]
        ans_type = entity_results[data_id]["ans_type"]
        attrs = list(entity_results[data_id]["attr"].split("|"))
        entity = entity_results[data_id]["entity"]
        answer = []
        for attr in attrs:
            f_ans = []
            temp_ans = []
            for kg in kg_trees:
                if attr != kg["attr"]: # 属性必须相同
                    continue
                constraint = kg["constraint"]
                for entities in entity_sysnoyms:
                    if entity in entities and kg["entity"] in entities: # 必须为同义实体
                        temp_ans = kg["ans"] # 属性值+实体匹配时，得到临时答案
                        is_f_ans = True if len(constraint) > 0 else False
                        for c in constraint:
                            if c not in query:
                                is_f_ans = False
                                break
                        if is_f_ans:
                            f_ans = kg["ans"] # 属性值+实体+约束属性值都匹配时，得到最终答案
                            break
                if f_ans != []:
                    break
            if f_ans == []:  # 没找到最终答案，用临时答案
                f_ans = temp_ans
            answer += f_ans
        answer = list(set(answer)) # set 去重
        if entity == "": # 没有实体，无答案
            answer = []
        results.append({
            "id": data_id,
            "query": query, 
            "ans_type": ans_type,
            "attr": "|".join(attrs),
            "entity": entity,
            "ans": "|".join(answer)
        })
                          
    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    with open(post_result_path, "w") as fp:
        fp.write(json_str)

def add_ans_from_triples_kg(pre_result_path, post_result_path, entity_synonyms_path, kg_path):
    # 当用训练集构造的kg没找到答案时，用triples中构造的kg填充没有找到答案的query
    entity_sysnoyms = load_synonyms(entity_synonyms_path)
    with open(pre_result_path) as fp:
        results = json.load(fp)
    with open(kg_path) as fp:
        kg_trees = json.load(fp)

    for result in results:
        if result["ans"] != "": # 已经找到答案
            continue
        attrs = list(result["attr"].split("|"))
        entity = result["entity"]
        if entity == "":
            continue
        query = result["query"]
        answer = []
        for attr in attrs:
            f_ans = []
            temp_ans =[]
            for kg in kg_trees:
                if attr != kg["attr"]:
                    continue
                constraint = kg["constraint"]
                for entities in entity_sysnoyms:
                    if entity in entities and kg["entity"] in entities:
                        temp_ans = kg["ans"] 
                        is_f_ans = True if len(constraint) > 0 else False
                        for c in constraint:
                            if c not in query:
                                is_f_ans = False
                                break
                        if is_f_ans:
                            f_ans = kg["ans"] 
                            break
                if f_ans != []:
                    break
            if f_ans == []:
                f_ans = temp_ans
            answer += f_ans
        answer = list(set(answer))
        result["ans"] = "|".join(answer)

    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    with open(post_result_path, "w") as fp:
        fp.write(json_str)

def update_ans_for_comp(pre_result_path, post_result_path, entity_synonyms_path, kg_path):
    entity_sysnoyms = load_synonyms(entity_synonyms_path)
    with open(pre_result_path) as fp:
        results = json.load(fp)
    with open(kg_path) as fp:
        kg_trees = json.load(fp)

    for result in results:
        if result["ans_type"] != "比较句":
            continue
        entity = result["entity"]
        if entity == "": # 没有实体，无答案
            continue
        attr = result["attr"]
        query = result["query"]
        f_ans = []
        temp_ans = []
        for kg in kg_trees:
            constraint = kg["constraint"]
            if kg["attr"] != attr:
                continue
            for entities in entity_sysnoyms:
                if kg["entity"] in entities and entity in entities:
                    temp_ans = kg["ans"]
                    is_f_ans = True if len(constraint) > 0 else False
                    for c in constraint:
                        if c not in query:
                            is_f_ans = True
                            break
                    if not is_f_ans:
                        f_ans = kg["ans"]
                        break
            if f_ans != []:
                break
        if f_ans == []:
            f_ans = temp_ans
        if f_ans == []:
            f_ans = ["no"] # 没有答案，用"no"作为默认值
        f_ans = list(set(f_ans))
        result["ans"] = "|".join(f_ans)

    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    with open(post_result_path, "w") as fp:
        fp.write(json_str)
 
def convert_json2csv(pre_result_path, post_result_path):
    with open(pre_result_path) as fp:
        json_obj = json.load(fp)
    df = pd.DataFrame(json_obj)
    df.to_csv(post_result_path)

def convert_for_upload(pre_result_path, post_result_path):
    with open(pre_result_path) as fp:
        results = json.load(fp)

    ret = dict()
    ret["result"] = dict()
    count1 = 0
    count2 = 0
    for result in results:
        data_id = result["id"]
        ans = result["ans"]
        ret["result"][data_id] = ans
        entity = result["entity"]
        if ans == "":
            count1 += 1
        if entity == "":
            count2 += 1
       
    print(f"{count1} samples's ans is none")
    print(f"{count2} samples's entity is none")
    
    json_str = json.dumps(ret, indent=4, ensure_ascii=False)
    with open(post_result_path, "w") as fp:
        fp.write(json_str)

def postprocess(path_env):
    postprocess_ans_type_result(path_env.ans_type_infer_result,
                                path_env.ans_type_result,
                                path_env.ans_type_kg_path)
    postprocess_attr_result(path_env.attr_infer_result,
                            path_env.attr_result,
                            path_env.infer_set_path,
                            path_env.ans_type_result,
                            path_env.attr_kg_path)

    extract_entity(path_env.attr_result,
                   path_env.entity_result,
                   path_env.entity_synonyms_path)

    generate_ans_from_train_kg(path_env.entity_result,
                               path_env.ans_from_train_tree,
                               path_env.entity_synonyms_path,
                               path_env.ans_tree_from_train_set)
    add_ans_from_triples_kg(path_env.ans_from_train_tree,
                            path_env.ans_from_train_triples_tree,
                            path_env.entity_synonyms_path,
                            path_env.ans_tree_from_triples)
    update_ans_for_comp(path_env.ans_from_train_triples_tree,
                        path_env.ans_processed_comp,
                        path_env.entity_synonyms_path,
                        path_env.ans_tree_for_compare)

    convert_json2csv(path_env.ans_processed_comp, path_env.final_result)
    convert_for_upload(path_env.ans_processed_comp, path_env.result_for_upload)

if __name__ == "__main__":
    path_env = FilePathConfig()
    postprocess(path_env)