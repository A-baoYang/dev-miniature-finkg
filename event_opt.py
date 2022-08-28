import json
import os
from time import time
from tqdm import tqdm
from KGBuilder.config import *
from KGBuilder.data_utils import *
from KGBuilder.event_utils import *


start = time()
# args["timestamp"] = 20220807
args["root_dir"] = os.getcwd()
input_event_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output", 
    # "entity-extraction-output-%s.json" % args["timestamp"]
    data_args["module_EventTri_output_filename"]
)
input_entity_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output", 
    # "entity-extraction-output-%s.json" % args["timestamp"]
    data_args["module_FiNER_output_filename"]
)
output_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output", 
    # "event-opt-output-%s.json" % args["timestamp"]
    data_args["module_EventOpt_output_filename"]
)
event_data = common_data_loader(filepath=input_event_filepath)
entity_data = common_data_loader(filepath=input_entity_filepath)


# 事件實體採比對 不重跑 NER 避免和整篇文章取出的實體不一
for i in tqdm(range(len(entity_data))):
    # event_triplets
    event_triplets = event_data[i]["event_triplets"]
    entities = entity_data[i]["entity_extract"]
    for event_set in event_triplets:
        entity_collect = []
        for eve_col in ["subject","verb","object","time","place"]:
            if f"event_{eve_col}" in event_set:
                target_text = event_set[f"event_{eve_col}"]
                for item in entities:
                    if item["entity"] in target_text:
                        entity_collect.append(item)

        if entity_collect:
            event_set.update({"entity_extract": entity_collect})
        else:
            event_set.update({"entity_extract": None})
    
    entity_data[i].update({"event_triplets": event_triplets})

data = entity_data.copy()
del event_data, entity_data
print(data[0])


for i in tqdm(range(len(data))):

    event_triplets = data[i]["event_triplets"]
    entities = data[i]["entity_extract"].copy()
    # 蒐集非 OTH 的實體
    entity_collect = collect_nonOTH_entity(entities)

    # 比對每個事件三元組所包含的實體數量決定是否過濾
    for event_set in event_triplets:
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

print("Duration: ", time() - start)
