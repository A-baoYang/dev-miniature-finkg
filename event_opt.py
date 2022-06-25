#%%
from collections import Counter
import json
import os
from tqdm import tqdm 
from KGBuilder.config import *
from KGBuilder.data_utils import *

args["root_dir"] = os.getcwd()
input_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output",
    "event_triplets-output-20220615.json"
    # "entity_extraction-output-20220614.json"
)
output_filepath = os.path.join(
    args["root_dir"], 
    args["model_dir"], "output",
    "event_filter-output-%s.json" % args["timestamp"]
)
data = common_data_loader(filepath=input_filepath)

#%%
# exclude events which both subject, object dont have any non-OTH entities
# - input: data
# - output: data (add filter_flag at every event in "event_triplets")
for i in tqdm(range(len(data))):

    event_triplets = data[i]["event_triplets"]
    entities = data[i]["entity_extract"]
    # 蒐集非 OTH 的實體
    entity_collect = []
    for item in [v for k, v in entities.items() if k != "OTH"]:
        entity_collect += item

    for event_set in event_triplets:
        # 比對如果 主詞和受詞都至少包含一個非 OTH 實體則 filter=1
        filter_flag = 1
        for j in range(len(event_set["event"])):
            if j != 1:
                _match = [chr for chr in entity_collect if chr in event_set["event"][j]]
                if not _match:
                    filter_flag = 0
        event_set.update({"filter_flag": filter_flag})


#%%
# keep origin context order
# - input: data(output of triplets)
# - output: events(id: event_triplet_list)
article = data[3]["article"]
# add id in event_triplets.py {0: ["xx","xx","xxx"], 1: ["xx","xxx","xxxx"], ...}
events = dict(enumerate(data[3]["event_triplets"]))

# event relations (contains)
# - input: events
# - output: eve_contain_eve
eve_contain_eve = {k: [] for k in range(len(events))}
for eid in events:
    event = events[eid]
    for _eid, _eval in events.items():
        if all(chr in "".join(event) 
               for chr in "".join(_eval)) and event != _eval:
            eve_contain_eve[eid].append(_eid)

eve_contain_eve
#%%
#  minimum event text
# - input: events, eve_contain_eve
# - output: text
exclude_ids = []
for k, v in eve_contain_eve.items():
    exclude_ids += v

text_order = []
for k, v in eve_contain_eve.items():
    if k not in exclude_ids:
        text_order.append(k)

text = []
for _id in text_order:
    text += events[_id] + ["\n"]
print("".join(text))


#%%
# with open(output_filepath, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)

