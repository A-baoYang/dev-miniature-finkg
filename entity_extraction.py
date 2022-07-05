import os
import json
import shutil
from tqdm import tqdm
from KGBuilder.config import *


# args["timestamp"] = 20220627
input_filepath = os.path.join(
    args["root_dir"], args["model_dir"], "output", # "event_triplets-output-%s.json" % args["timestamp"]
    data_args["module_FiNER_input_filename"]
)
output_filepath = os.path.join(
    args["root_dir"], args["model_dir"], "output", # "entity_extraction-output-%s.json" % args["timestamp"]
    data_args["module_FiNER_output_filename"]
)
# 將 FiNER 所需的 slot_label.txt 從 dictionary 資料夾移過來
filepath = os.path.join(args["root_dir"], args["model_dir"], "input", "slot_label.txt")
if not os.path.exists(filepath):
    shutil.copyfile(
        os.path.join(args["root_dir"], gcs_args["gcs_dict_directory"], "slot_label.txt"), 
        filepath
    )


# CKIP Tagger || CKIP Transformers
from KGBuilder.ckipNER.ckip import CKIPWrapper
ckip = CKIPWrapper()
input_data = ckip.data_loader(input_filepath=input_filepath)
for i in tqdm(range(len(input_data))):
    # article
    input_text = input_data[i]["article"]
    _res_ckip = ckip.ner_predict(input_text=[input_text])[0]["entity_text"]
    input_data[i].update({"entity_extract": _res_ckip})


# FiNER
from KGBuilder.FiNER.trainer import FiNER
from KGBuilder.FiNER.utils import (
    load_tokenizer,
    set_seed,
)

# 隨機亂數 & 模型參數初始化 / 載入 bert tokenizer & FiNER model
set_seed()
tokenizer = load_tokenizer()
finer = FiNER()
finer.load_model()
# 開始預測實體種類
for i in tqdm(range(len(input_data))):
    # article
    text_tokens = " ".join([word for word in input_data[i]["article"]])
    _res_finer = finer.predict([text_tokens], tokenizer)[0]["entity_text"]
    _res_finer["EVENT"] = _res_finer.pop("EVE")
    _res_finer["PRODUCT"] = _res_finer.pop("GOO")
    for ent_type in _res_finer.keys():
        if ent_type in input_data[i]["entity_extract"]:
            input_data[i]["entity_extract"][ent_type] += _res_finer[ent_type]
            input_data[i]["entity_extract"][ent_type] = list(set(input_data[i]["entity_extract"][ent_type]))
        else:
            input_data[i]["entity_extract"].update({ent_type: _res_finer[ent_type]})
        

# 事件實體採比對 不重跑 NER 模型因為和整篇文章取出的實體不一)
for i in tqdm(range(len(input_data))):
    # event_triplets
    entities = input_data[i]["entity_extract"]
    event_triplets = input_data[i]["event_triplets"]
    for i, event_set in event_triplets.items():
        entity_collect = {}
        for j in range(len(event_set["event"])):
            target_text = event_set["event"][j]
            for _type, ent_list in entities.items():
                _match = [ent for ent in ent_list if ent in target_text]
                if _match:
                    entity_collect.update({_type: _match})

        if entity_collect:
            event_set.update({"entity_extract": entity_collect})
        else:
            event_set.update({"entity_extract": None})


with open(output_filepath, "w", encoding="utf-8") as f:
    json.dump(input_data, f, ensure_ascii=False, indent=4)
