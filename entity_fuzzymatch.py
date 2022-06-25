from fuzzychinese import FuzzyChineseMatch  # https://github.com/znwang25/fuzzychinese
import numpy as np
from opencc import OpenCC
tw2sp = OpenCC("tw2sp")
s2twp = OpenCC("s2twp")
import os
import pandas as pd

from KGBuilder.config import *
from KGBuilder.data_utils import *


input_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output",
    data_args["module_FiNER_output_filename"]
)
output_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output",
    "entity_fuzzymatch-rel-output-20220614.csv"
)
data = common_data_loader(filepath=input_filepath)

# 實體
entities_org = []
for item in data:
    if "GPE" in item["entity_extract"]:
        entities_org += item["entity_extract"]["GPE"]
entities_org = [item.strip() for item in np.unique(entities_org) if item.strip()]
print(len(entities_org))

# 關係觸發詞
rel_trigger = []
for item in data:
    for _set in item["event_triplets"]:
        rel_trigger.append(_set["event"][1])
rel_trigger = [item.strip() for item in np.unique(rel_trigger) if item.strip()]
print(len(entities_org))



n = 5
ngram_range = (3, 3)
fuzzy_options = pd.Series(rel_trigger)
fcm = FuzzyChineseMatch(ngram_range=ngram_range, analyzer="stroke")
fcm.fit(fuzzy_options)
top_sim = fcm.transform(rel_trigger, n=n)


res = pd.concat([
    fuzzy_options,
    pd.DataFrame(top_sim, columns=[f"top{i}" for i in range(1, n+1)]),
    pd.DataFrame(
        fcm.get_similarity_score(),
        columns=[f'top{i}_score' for i in range(1, n+1)])],
                axis=1)

res = res.rename(columns={0: "origin_name"})
res = res.drop(["top1", "top1_score"], axis=1)
res.to_csv(output_filepath, index=False)
