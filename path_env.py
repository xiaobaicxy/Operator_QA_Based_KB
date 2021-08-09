import os

class FilePathConfig:
    def __init__(self):
        # 原始文件
        self.train_set_path = "./dataset/origin/train.xlsx"
        self.infer_set_path = "./dataset/origin/test1.xlsx"
        self.entity_synonyms_path = "./dataset/origin/synonyms.txt"
        self.triples_path = "./dataset/origin/triples.txt"

        # 训练集qa对
        self.ans_type_qa_pair_path = "./dataset/post/ans_type_qa_pair_path.json"
        self.attr_qa_pair_path = "./dataset/post/attr_qa_pair_path.json"
        if not os.path.exists("./dataset/post/"):
            os.makedirs("./dataset/post/")

        # 模型路径
        self.ans_type_model = "./models/ans_type_model_parameter.pkl"
        self.attr_model = "./models/attr_model_parameter.pkl"
        if not os.path.exists("./models"):
            os.makedirs("./models")

        # 字典
        self.vocab_path = "./dataset/vocab/vocab.txt"
        if not os.path.exists("./dataset/vocab"):
            os.makedirs("./dataset/vocab")

        # 知识树
        self.ans_type_kg_path = "./dataset/knowledges/ans_type_candidates.json"
        self.attr_kg_path = "./dataset/knowledges/attr_candidates.json"
        self.ans_tree_from_train_set = "./dataset/knowledges/ans_tree_from_train_set.json"
        self.ans_tree_from_triples = "./dataset/knowledges/ans_tree_from_triples.json"
        self.ans_tree_for_compare = "./dataset/knowledges/ans_tree_for_compare.json"
        if not os.path.exists("./dataset/knowledges/"):
            os.makedirs("./dataset/knowledges/")

        # infer结果
        self.ans_type_infer_result = "./results/infer_results/ans_type_result.json"
        self.attr_infer_result = "./results/infer_results/attr_result.json"
        if not os.path.exists("./results/infer_results/"):
            os.makedirs("./results/infer_results/")

        # 结果后处理
        self.ans_type_result = "./results/post_results/ans_type_result.json"
        self.attr_result = "./results/post_results/attr_result.json"
        self.entity_result = "./results/post_results/entity_result.json"
        self.ans_from_train_tree = "./results/post_results/ans_from_train_tree.json"
        self.ans_from_train_triples_tree = "./results/post_results/ans_from_train_triples_tree.json"
        self.ans_processed_comp = "./results/post_results/ans_processed_comp.json"
        if not os.path.exists("./results/post_results/"):
            os.makedirs("./results/post_results/")
        
        # 提交文件
        self.final_result = "./results/final_results/result.csv"
        self.result_for_upload = "./results/final_results/kbqa_results.json"
        if not os.path.exists("./results/final_results"):
            os.makedirs("./results/final_results")
