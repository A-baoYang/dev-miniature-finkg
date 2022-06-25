import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import logging
import shutil
from tqdm import tqdm

from KGBuilder.config import *


args["root_dir"] = os.getcwd()
gcs_args[
    "service_account_key"
] = "../../forResearch/data/gcs/dst-dev2021-c63c2e59ebbb.json"


# 從 GCS 下載模型至本地端(原始的 bert-base-chinese 以及 已訓練好的 FiNER & FinRE)
from KGBuilder.data_utils import gcs_downloader

for model_dir in ["baseBert", "FiNER", "FinRE"]:
    download_to_dir = gcs_args[f"local_model_dir_{model_dir}"]
    source_dir = os.path.join(gcs_args["gcs_model_directory"], download_to_dir)
    gcs_downloader(source_dir, download_to_dir)

# 將 FiNER 所需的 slot_label.txt 從 dictionary 資料夾移過來
filename = "slot_label.txt"
if not os.path.exists(os.path.join(args["model_dir"], "input", filename)):
    shutil.copyfile(
        os.path.join(gcs_args["gcs_dict_directory"], filename), 
        os.path.join(args["model_dir"], "input", filename)
    )

# Start prediction
# CKIP Tagger || CKIP Transformers
from KGBuilder.OpenModels.args import Args
args_ckip = Args()
from KGBuilder.OpenModels.ckip import CKIPWrapper
ckip = CKIPWrapper()
ner_results = ckip.ner_predict(input_filepath="cleansed-quick-technews-俄烏戰爭.csv")


# FiNER
from KGBuilder.NamedEntityRecognition.args import Args
args_finer = Args()
from KGBuilder.NamedEntityRecognition.trainer import FiNER
from KGBuilder.NamedEntityRecognition.utils import (
    init_logger,
    load_tokenizer,
    read_prediction_text,
    set_seed,
)

logger = logging.getLogger(__name__)
args_finer.do_pred = True  # 指定模式為：預測
args_finer.use_crf = True  # 指定使用 CRF layer
args_finer.task = "input"
args_finer.model_dir = "FiNER"
args_finer.data_dir = "./model_data"
# args_finer.finer_pred_input_file = "model-input-LTN-%s.json" % (args["timestamp"])
args_finer.finer_pred_input_dir = "./article_data/"
args_finer.finer_pred_input_file = "cleansed-quick-technews-俄烏戰爭.csv"
args_finer.finer_pred_output_file = "FinKG-NER-%s.json" % (args["timestamp"])
init_logger(args_finer)  # logger 初始化
set_seed(args_finer)  # 隨機亂數初始化
tokenizer = load_tokenizer(args_finer)  # 載入 bert tokenizer (original BERT)
finer = FiNER(args_finer)  # 參數初始化
finer.load_model()  # 載入 FiNER 模型
# 將輸入文本轉為 embeddings
text_tokens = read_prediction_text(
    pred_input_dir=args_finer.finer_pred_input_dir,
    pred_input_file=args_finer.finer_pred_input_file
)
ner_results = finer.predict(text_tokens, tokenizer)  # 儲存 NER 預測(抽取)結果並回傳
print(ner_results[0])


# Merge NER results
with open(os.path.join(args_ckip.ckip_pred_output_dir, args_ckip.ckip_pred_output_file), "r", encoding="utf-8") as f:
    ckip_ner_results = json.load(f)
with open(os.path.join(args_finer.finer_pred_output_dir, args_finer.finer_pred_output_file), "r", encoding="utf-8") as f:
    finer_ner_results = json.load(f)
assert len(ckip_ner_results) == len(finer_ner_results)

merged_ner_results = []
for _id in tqdm(range(len(finer_ner_results))):
    text = ckip_ner_results[_id]["text"]
    ckip_ner = ckip_ner_results[_id]["entity_text"]
    finer_ner = finer_ner_results[_id]["entity_text"]
    finer_ner["PRODUCT"] = finer_ner.pop("GOO")
    finer_ner["EVENT"] = finer_ner.pop("EVE")
    intersect_ent_types = list(set(ckip_ner.keys()).intersection(set(finer_ner.keys())))
    union_ent_types = list(set(ckip_ner.keys()).union(set(finer_ner.keys())))
    _merge_dict = {}
    for ent_type in union_ent_types:
        if ent_type in intersect_ent_types:
            ent_names = list(set(ckip_ner[ent_type]).union(set(finer_ner[ent_type])))
        else:
            if ent_type in ckip_ner.keys():
                ent_names = ckip_ner[ent_type]
            else:
                ent_names = finer_ner[ent_type]
        _merge_dict.update({ent_type: ent_names})
    merged_ner_results.append({"text": text, "entity_text": _merge_dict})

with open(
    "./model_data/output/merged_ner_results-%s.json" % args["timestamp"], 
    "w", 
    encoding="utf-8"
) as f:
    json.dump(merged_ner_results, f, ensure_ascii=False, indent=4)


# # FinRE
# # 載入模型要預測的 input data
# print("====== Load Data ======")
# filepath = os.path.join(
#     args["model_dir"], "input", data_args["model_FinRE_input_filename"]
# )
# finre_input = json.load(open(filepath, encoding="utf-8"))
# print(f"Length of FinRE data to predict: {len(finre_input)}")
# print(f"Sample data: \n{finre_input[0]}")

# from KGBuilder.RelationExtraction.predict import FinRE
# # Start Prediction
# finre = FinRE(local_model_directory=gcs_args["local_model_directory_FinRE"])  # 載入 FinRE 模型
# result, result_json = finre.predict(data=finre_input)  # 進行預測
# print(result.head())
# print(result_json[0])
