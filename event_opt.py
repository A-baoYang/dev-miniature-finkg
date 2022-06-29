import json
import os
from tqdm import tqdm
from KGBuilder.config import *
from KGBuilder.data_utils import *
from KGBuilder.event_utils import *


# args["timestamp"] = 20220627
args["root_dir"] = os.getcwd()
input_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output", 
    # "entity_extraction-output-%s.json" % args["timestamp"]
    data_args["module_FiNER_output_filename"]
)
output_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output", 
    # "event_opt-output-%s.json" % args["timestamp"]
    data_args["module_EventOpt_output_filename"]
)
data = common_data_loader(filepath=input_filepath)


for i in tqdm(range(len(data))):

    event_triplets = data[i]["event_triplets"]
    entities = data[i]["entity_extract"].copy()
    # 蒐集非 OTH 的實體
    entity_collect = collect_nonOTH_entity(entities)

    # 比對每個事件三元組所包含的實體數量決定是否過濾
    for _id, event_set in event_triplets.items():
        filter_flag = filter_if_contains_nonOTH_entity(event_set, entity_collect)
        event_set.update({"filter_flag": filter_flag})

    # 識別該篇文章所有事件間的嵌套關係
    eve_contain_eve = check_nested_relations(events=event_triplets)
    # 根據嵌套關係，輸出該篇文章的不重複事件文字
    minimum_text = form_minimum_text(event_rels=eve_contain_eve, events=event_triplets)

    data[i].update({
        "event_relations": eve_contain_eve,
        "minimum_event": minimum_text
    })


with open(output_filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
