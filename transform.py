import argparse
import json
import os
import pandas as pd
from KGBuilder.config import *
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--module', type=str, default="EventTri")
parser_args = parser.parse_args()
module = parser_args.module
input_filepath = os.path.join(args["root_dir"], args["article_dir"], data_args["cleansed_filename"])
output_filepath = os.path.join(args["root_dir"], args["model_dir"], "input", data_args[f"module_{module}_input_filename"])


# 若無 model_data/input/ & model_data/output/ 資料夾則先創建
for postfix in ["input", "output"]:
    path = os.path.join(args["root_dir"], args["model_dir"], postfix)
    Path(path).mkdir(parents=True, exist_ok=True)


# 從 GCS 下載模型至本地端(原始的 bert-base-chinese)
from KGBuilder.data_utils import gcs_downloader
download_to_dir = os.path.join(args["root_dir"], gcs_args["baseBert"])
source_dir = os.path.join(gcs_args["gcs_model_directory"], gcs_args["baseBert"])
gcs_downloader(source_dir, download_to_dir)


# 根據模組參數進行相對應的資料格式轉換
df = pd.read_csv(input_filepath)
if module == "EventTri":
    df["article"] = df["article"].fillna("")
    df["article"] = df.apply(lambda row: row["title"]+row["article"], axis=1)
    df = df[["title","url","meta_keywords","article","publish_time","media"]]
    df = df[df["media"] != "MoneyDJ"].reset_index(drop=True)
    print(df.shape)
    output = df.to_dict(orient="records")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
else:
    print("module not set")
