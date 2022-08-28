import json
import os
from time import time
from tqdm import tqdm
from KGBuilder.config import *


start = time()
# args["timestamp"] = 20220807
input_filepath = os.path.join(
    args["root_dir"], args["model_dir"], "input", # "event-triplets-input-%s.json" % args["timestamp"]
    data_args["module_EventTri_input_filename"]
)
output_filepath = os.path.join(
    args["root_dir"], args["model_dir"], "output", # "event-triplets-output-%s.json" % args["timestamp"]
    data_args["module_EventTri_output_filename"]
)

# 載入資料
with open(input_filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

# 對文本進行基於句法分依存分析的事件三元組標註
from KGBuilder.EventTriplet.ltp import TripleExtractor, CausalityExractor
triplet_extractor = TripleExtractor(model_tag="base2")
causality_extractor = CausalityExractor(model_tag="base2")

for i in tqdm(range(len(data))):
    try:
        triplets = [
            {
                "event_subject": eve[0], "event_verb": eve[1], "event_object": eve[2], 
                "event_time": eve[3], "event_place": eve[4], 
                "event_type": "triplet"
            }
            for eve in triplet_extractor.triples_main(data[i]["article"])
            if len(eve) == 5
        ]
    except:
        triplets = []

    causalities = []
    try:
        for _cau in causality_extractor.extract_main(data[i]["article"]):
            cause = "".join([word.split("/")[0] for word in _cau["cause"].split(" ") if word.split("/")[0]])
            tag = _cau["tag"]
            effect = "".join([word.split("/")[0] for word in _cau["effect"].split(" ") if word.split("/")[0]])
            causalities.append({
                "event_subject": cause, "event_verb": tag, "event_object": effect,
                "event_type": "causality"
            })
    except:
        continue

    _res = []
    for id, item in enumerate(triplets + causalities):
        item.update({"event_no": id})
        _res.append(item)
    data[i].update({"event_triplets": _res})

# 儲存輸出
with open(output_filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Duration: ", time() - start)