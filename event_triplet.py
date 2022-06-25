import os
import json
from tqdm import tqdm
from KGBuilder.config import *


input_filepath = os.path.join(
    args["root_dir"], args["model_dir"], "input", data_args["module_EventTri_input_filename"]
)
output_filepath = os.path.join(
    args["root_dir"], args["model_dir"], "output", data_args["module_EventTri_output_filename"]
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
        triplets = triplet_extractor.triples_main(data[i]["article"])
    except:
        triplets = []

    _causalities = causality_extractor.extract_main(data[i]["article"])
    causalities = []
    try:
        for _cau in _causalities:
            cause = ''.join([word.split('/')[0] for word in _cau['cause'].split(' ') if word.split('/')[0]])
            tag = _cau["tag"]
            effect = ''.join([word.split('/')[0] for word in _cau['effect'].split(' ') if word.split('/')[0]])
            causalities.append([cause, tag, effect])
    except:
        continue

    data[i].update({"event_triplets": triplets + causalities})


# 儲存輸出
with open(output_filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
