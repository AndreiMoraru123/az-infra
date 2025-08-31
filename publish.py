import argparse
import os
import shutil


def copy_all(src: str, dst: str):
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--eval_results", type=str)
    parser.add_argument("--final_output", type=str)
    args = parser.parse_args()

    os.makedirs(args.final_output, exist_ok=True)

    copy_all(args.model_path, os.path.join(args.final_output, "model"))
    copy_all(args.eval_results, os.path.join(args.final_output, "eval"))
