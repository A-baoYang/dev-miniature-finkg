import json
import os
from tqdm import tqdm
from KGBuilder.config import *
from KGBuilder.data_utils import *
from KGBuilder.EntityMerge.entity_utils import *
from KGBuilder.EntityMerge.merger import SimilarFinder


args["root_dir"] = os.getcwd()
args["timestamp"] = 20220627
input_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output",
    # data_args["module_EventOpt_output_filename"]
    "event_opt-output-%s.json" % args["timestamp"]
)
output_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output",
    # data_args["module_EntityMerge_output_filename"]
    "similar_entity_dict-%s.json" % args["timestamp"]
)
vector_npyfilepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output",
    data_args["module_EntityVector_filename"]
)
# 變數初始化
data = common_data_loader(filepath=input_filepath)
similar_finder = SimilarFinder()
# 獲取不重複實體列表
entity_types = get_unique_entities(data)

# 使用 BERT 推測事件向量，再切出實體向量
# 若有舊的實體向量會一併載入，並更新本地向量儲存檔
all_entity_vectors = similar_finder.get_entity_embeddings(
    model_name="hfl/chinese-roberta-wwm-ext-large", 
    data=data,
    entity_types=entity_types
)

entity_map = dict()
for entity_type in tqdm(entity_types):
    
    # 轉換為簡中輸入、排除包含英文的實體（fuzzychinese 無法進行比對故在此排除）
    entities = []
    for item in data:
        if entity_type in item["entity_extract"]:
            entities += similar_finder.to_simplified_chinese(word_list=item["entity_extract"][entity_type])
    entities = [item for item in list(set(entities)) if item]

    # 對每個實體類型跑過相似實體匹配
    # (1) 使用 fuzzychinese
    fuzzy_res = similar_finder.match_by_stroke(
        sim_num=2, entities=entities, 
        ngram_range=(3, 3), analyzer="stroke"
    )
    fuzzy_res["entity_type"] = entity_type

    # (2) 使用 BERT Embeddings
    bert_res = similar_finder.match_by_embeddings(
        all_entity_vectors=all_entity_vectors, 
        entity_type=entity_type,
        distance_metric="euclidean"
    )
    # 合併 (1) 和 (2) 結果
    res = pd.concat([fuzzy_res, bert_res], axis=0)

    # 篩選 (1) & (2) 交集（尋找到最相似詞一致）存為別名同義詞庫
    _cond = res.groupby(["origin_entity","top1"]).agg({"top1_metric": "count"}).reset_index()
    _select_entities = _cond[_cond["top1_metric"] > 1]["origin_entity"].values.tolist()
    df = res[res["origin_entity"].isin(_select_entities)].sort_values(["entity_type","origin_entity"])
    _tmp_entity_map = df.iloc[:, :2].set_index("origin_entity").to_dict()["top1"]

    # 轉為 key-list pair、過濾重複詞彙
    _entity_map = {}
    for k, v in _tmp_entity_map.items():
        if v not in _entity_map and v not in _entity_map.values():
            _entity_map.update({k: v})
    _entity_map = {k: [k, v] for k, v in _entity_map.items()}
    entity_map.update({entity_type: _entity_map})


with open(output_filepath, "w", encoding="utf-8") as f:
    json.dump(entity_map, f, ensure_ascii=False, indent=4)
