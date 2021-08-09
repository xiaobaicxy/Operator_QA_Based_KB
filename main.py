from config import Config
from path_env import FilePathConfig
from generate_kg import generate_kg
from generate_qa_pair import generate_qa_pair
from generate_vocab import generate_vocab
from run_model import run_model
from postprocessor import postprocess

def main(path_env, config):
    print("###### generate kg begin ######")
    generate_kg(path_env)
    print("###### generate kg end ######")

    print("###### generate qa pair begin ######")
    generate_qa_pair(path_env)
    print("###### generate qa pair end ######")
    
    print("###### generate vocab begin ######")
    generate_vocab(path_env)
    print("###### generate vocab end ######")

    run_params = [("train", "ans_type"), ("train", "attr"), ("infer", "ans_type"), ("infer", "attr")]
    for run_type, data_type in run_params:
        print(f"###### {run_type} {data_type} begin ######")
        run_model(path_env, config, run_type, data_type)
        print(f"###### {run_type} {data_type} end ######")
    
    print("###### postprocess begin ######")
    postprocess(path_env)
    print("###### postprocess end ######")

if __name__ == "__main__":
    path_env = FilePathConfig()
    config = Config()
    main(path_env, config)
    