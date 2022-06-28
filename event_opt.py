from collections import Counter
import json
import os
from tqdm import tqdm
from KGBuilder.config import *
from KGBuilder.data_utils import *

args["root_dir"] = os.getcwd()
input_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output", # "entity_extraction-output-20220626.json"
    data_args["module_FiNER_output_filename"]
)
output_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output", # "event_opt-output-20220626.json"
    data_args["module_EventOpt_output_filename"]
)
data = common_data_loader(filepath=input_filepath)


def collect_nonOTH_entity(entities):
    """ 
    蒐集非 OTH 的實體 
    - entities: 從該篇文章抽取出的所有實體
    """
    entity_collect = []
    for item in [v for k, v in entities.items() if k != "OTH"]:
        entity_collect += item
    return entity_collect


def filter_if_contains_nonOTH_entity(event_set, entity_collect):
    """ 
    比對如果 主詞和受詞都至少包含一個非 OTH 實體則 filter=1 反之則 filter=0
    - event_set: 該篇文章下的其中一個事件三元組抽取結果
    - entity_collect: 該篇文章除了 `OTH` 類型以外的所有實體
    """
    filter_flag = 1
    for j in range(len(event_set["event"])):
        if j != 1:
            _match = [chr for chr in entity_collect if chr in event_set["event"][j]]
            if not _match:
                filter_flag = 0
    return filter_flag


def check_nested_relations(events):
    """
    識別事件嵌套關係
    [Input]
    - events: 該篇文章的所有事件
    [Output]
    - eve_contain_eve: 該篇文章的事件嵌套關係
    """
    eve_contain_eve = {}
    for eid, event_item in events.items():
        _contain_eve = []
        for _eid, _eval in events.items():
            if all(chr in "".join(event_item["event"]) for chr in "".join(_eval["event"])) and event_item["event"] != _eval["event"]:
                _contain_eve.append(_eid)
        if _contain_eve:
            eve_contain_eve[eid] = _contain_eve
    return eve_contain_eve


def form_minimum_text(event_rels, events):
    exclude_ids = []
    for k, v in event_rels.items():
        exclude_ids += v

    text_order = []
    for k, v in events.items():
        if k not in exclude_ids:
            text_order.append(k)

    text = []
    for _id in text_order:
        text += events[_id]["event"] + ["\n"]
    return "".join(text)


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
