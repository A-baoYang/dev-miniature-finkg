#%%
from fuzzychinese import FuzzyChineseMatch
import json
from opencc import OpenCC
import os
from tqdm import tqdm
from KGBuilder.config import *
from KGBuilder.data_utils import *
import warnings
warnings.filterwarnings("ignore")


args["root_dir"] = os.getcwd()
args["timestamp"] = 20220626
input_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output",
    "event_opt-output-%s.json" % args["timestamp"]
)
output_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output",
    "similar_entity_table-%s.json" % args["timestamp"]
)
data = common_data_loader(filepath=input_filepath)

# %%
# 獲取不重複實體列表
entity_types = []
for item in data:
    for type in item["entity_extract"]:
        if type not in entity_types:
            entity_types.append(type)

#%%
# 對每個實體類型跑過相似實體匹配
tw2sp = OpenCC("tw2sp")
s2twp = OpenCC("s2twp")

# for event_type in tqdm(entity_types):
event_type = entity_types[0]
entities = []
for item in data:
    if event_type in item["entity_extract"]:
        entities += item["entity_extract"][event_type]
entities = [tw2sp.convert(item.strip()) for item in list(set(entities)) if item.strip()]

fuzzy_options = pd.Series(entities)
fcm = FuzzyChineseMatch(ngram_range=(3, 3), analyzer="stroke")
fcm.fit(fuzzy_options)
top_sim = fcm.transform(entities, n=2)
res = pd.concat([
    fuzzy_options,
    pd.DataFrame(top_sim, columns=[f"top{i}" for i in range(1, n+1)]),
    pd.DataFrame(
        fcm.get_similarity_score(),
        columns=[f'top{i}_score' for i in range(1, n+1)])],
                axis=1)

res = res.rename(columns={0: "origin_name"})
res = res.drop(["top1", "top1_score"], axis=1)



# %%
