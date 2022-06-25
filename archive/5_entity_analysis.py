#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from KGBuilder.config import *
from KGBuilder.neo4j import *
from sentiment_ana_n_merge import file_io

args["root_dir"] = os.getcwd()
args["timestamp"] = "20220425-step6"
input_filepath = "model_data/output/merged_all-%s.json" % args["timestamp"]

#%%
# 通用 py2neo function
class Neo4jStore(object):
    def __init__(self, host, username, password, database):
        """
        連接 Neo4j 資料庫
        """
        link = Graph(host=host, auth=(username, password), name=database)
        self.graph = link
        self.matcher = NodeMatcher(self.graph)
    
    def create_node(self, node):
        """
        創建節點；若相同節點已存在則不創建，避免重複節點產生
        """
        name = node["name"]
        type = node["type"]
        if not self.matcher.match(type, name=name).first():
            name_node = Node(type, name=name)
            for k in node.keys():
                if k not in ["name", "type"]:
                    name_node[k] = node[k]
            self.graph.create(name_node)

    def create_relation(self, rel_set):
        ent1 = rel_set[0]
        rel = rel_set[1]
        ent2 = rel_set[2]
        if ent1["type"] != "NEWS":
            relationship = Relationship(
                self.matcher.match(ent1["type"], name=ent1["name"]).first(),
                rel["name"],
                self.matcher.match(ent2["type"], name=ent2["name"]).first(),
            )
        else:
            relationship = Relationship(
                self.matcher.match(ent1["type"], id=ent1["id"]).first(),
                rel["name"],
                self.matcher.match(ent2["type"], name=ent2["name"]).first(),
            )
        if not relationship:
            for k in rel.keys():
                if k != "name":
                    relationship[k] = rel[k]
            self.graph.create(relationship)
        else:
            self.graph.push(relationship)
            self.graph.commit()


#%%
# 載入 merged_all.json
merge_all_data = file_io(method="read", filepath=input_filepath, newline_json=False)

#%%
# 對文本進行基於句法分依存分析的事件抽取
# LTP: seg, pos, ner, srl, dep, sdp
from KGBuilder.OpenModels.ltp import TripleExtractor, CausalityExractor
extractor = TripleExtractor()
# content2 = """烏克蘭供應全球半數氖氣，現因俄烏戰事恐斷供。專家分析，台韓等半導體廠早採分散供應策略、影響面小，但歐美中小型半導體整合元件廠（IDM）的曝險程度較高，恐讓物聯網（IoT）和車用電子晶片出現長短料狀況。路透社報導，烏國兩大氖氣供應商Ingas和Cryoin提供全球約50%半導體主要光源原料，客戶包括台灣、韓國、中國、美國和德國等半導體公司，現因俄烏戰事被迫停產。對此，台灣經濟部回應說，台灣企業有安全庫存，短期內供應無虞。工研院產業科技國際策略發展所研究總監楊瑞臨今天下午接受中央社記者電訪說，氖氣是生產鋼鐵後的副產品，俄羅斯鋼鐵廠多數設有氖氣回收設備，但回收的氖氣雜質偏高，因此送往烏克蘭進行工業等級純化作業，再供應給全球主要半導體混合氣體廠商，進行半導體等級的純化作業。他指出，亞洲半導體廠在晶圓先進製程採用深紫外光（DUV）曝光設備，所需氖氣與其他特殊混合氣體的需求量大；2014年俄羅斯與烏克蘭因克里米亞（Crimean）爆發衝突，台灣與韓國廠商就率先對特殊氣體，採取分散風險彈性化與貨源供應多元性措施，以降低曝險程度。至於美國和歐洲的中小型半導體IDM廠，楊瑞臨表示，其晶圓製造以傳統和成熟製程為主，絕大部分採DUV設備，但半導體等級混合氣體需求量較亞洲廠商少；因此，2014年克里米亞危機後，並未採取氣體原料分散風險措施。面對國際氖氣價格飆漲，楊瑞臨分析，亞洲半導體廠商因使用量大，具有價格談判實力；歐美中小型IDM廠基於對烏克蘭氖氣原料依賴度高，加上使用量較少、難有價格談判優勢，導致曝險程度增加。他還指出，亞洲半導體廠商進入7奈米以下先進製程階段，逐步採用極紫外光（EUV）曝光機台，EUV設備使用二氧化碳氣體雷射、不用氖氣；亦即，更高階晶圓製程不受氖氣斷供影響。楊瑞臨認為，需留意歐美中小型IDM廠在半導體等級混合氣體的安全庫存狀況，若俄烏戰事延長，不排除歐美中小型IDM廠採產能降載措施因應，恐影響物聯網（IoT）和車用電子晶片出現長短料市況，車用晶片荒是否更嚴峻、值得觀察。（作者：鍾榮峰；首圖來源：shutterstock）"""

for i in tqdm(range(len(merge_all_data))): 
    svos = extractor.triples_main(merge_all_data[i]["article"])
    merge_all_data[i].update({"event_triplets": svos})

#%%
# 識別實體落點、建立實體關係
# 結果變得太分散，改成先事件圖譜、再從事件拉出提及實體的關係
for i in tqdm(range(len(merge_all_data))):
    entitiess = []
    for l in list(merge_all_data[i]["entities"].values()):
        entitiess.extend(l)
    eve_triplets = merge_all_data[i]["event_triplets"].copy()
    # rels = []
    for eve_id in tqdm(range(len(eve_triplets))):
        heads, tails = [], []
        for ent in entitiess:
            ent_type = [k for k, v in merge_all_data[i]["entities"].items() if ent in v][0]
            if ent in eve_triplets[eve_id][0]:
                heads.append((ent, ent_type))
            elif ent in eve_triplets[eve_id][2]:
                tails.append((ent, ent_type))
        eve_triplets[eve_id] = [
            {"event_head": eve_triplets[eve_id][0], "mentioned": heads},
            {"event_rel": eve_triplets[eve_id][1]},
            {"event_tail": eve_triplets[eve_id][2], "mentioned": tails}
        ]
    merge_all_data[i].update({"entity_eve_mentions": eve_triplets})
    merge_all_data[i]["event_triplets"] = [", ".join(item) for item in merge_all_data[i]["event_triplets"]]

    #     for h in heads:
    #         for t in tails:
    #             rels.append((h, eve_tri[1], t))
    # merge_all_data[i].update({"entity_eve_relations": rels})
    

# %%
# 基於詞組規則的因果關係抓取
extractor = CausalityExractor()
for i in tqdm(range(len(merge_all_data))):
    datas = extractor.extract_main(merge_all_data[i]["article"])
    _res = []
    for data in datas:
        cause = ''.join([word.split('/')[0] for word in data['cause'].split(' ') if word.split('/')[0]])
        tag = data["tag"]
        effect = ''.join([word.split('/')[0] for word in data['effect'].split(' ') if word.split('/')[0]])
        _res.append((cause, tag, effect))
    merge_all_data[i].update({"event_causality": _res})

    entitiess = []
    for l in list(merge_all_data[i]["entities"].values()):
        entitiess.extend(l)
    _res = merge_all_data[i]["event_causality"].copy()
    for eve_id in tqdm(range(len(_res))):
        heads, tails = [], []
        for ent in entitiess:
            ent_type = [k for k, v in merge_all_data[i]["entities"].items() if ent in v][0]
            if ent in _res[eve_id][0]:
                heads.append((ent, ent_type))
            elif ent in _res[eve_id][2]:
                tails.append((ent, ent_type))
        _res[eve_id] = [
            {"reason": _res[eve_id][0], "mentioned": heads},
            {"rel": _res[eve_id][1]},
            {"result": _res[eve_id][2], "mentioned": tails}
        ]
    merge_all_data[i].update({"event_causality_ent_mentions": _res})
    merge_all_data[i]["event_causality"] = [", ".join(item) for item in merge_all_data[i]["event_causality"]]

#%%
file_io(method="write", filepath=input_filepath, data=merge_all_data, newline_json=False)

# %%
# 轉存至 Neo4j
host = account_args["neo4j_hostname"] = "10.0.10.6"
username = account_args["neo4j_username"] = "neo4j"
password = account_args["neo4j_password"] = "neo4jj"
database = account_args["neo4j_database"] = "eventkg"
neo4j = Neo4jStore(host, username, password, database)

for _id, _set in tqdm(enumerate(merge_all_data)):
    # 節點&關係創建: event_triplets(entity_eve_mentions)
    try:
        news = {
            "type": "NEWS", "name": _set["title"], 
            "publish_time": _set["publish_time"], 
            "id": "%s-%s" % (_set["publish_time"], str(_id).zfill(5))
        }
        for idx in range(len(_set["entity_eve_mentions"])):
            eve_tri = _set["entity_eve_mentions"][idx]
            eve_tfidf, eve_sent = _set["event_triplets"][idx][1], _set["event_triplets"][idx][2]
            head = {"type": "EVENT_HEAD", "name": eve_tri[0]["event_head"]}
            tail = {"type": "EVENT_TAIL", "name": eve_tri[2]["event_tail"]}
            neo4j.create_node(node=head)
            neo4j.create_node(node=tail)
            rel = {"name": "涉及"}
            neo4j.create_relation(rel_set=(news, rel, head))
            rel = {"name": eve_tri[1]["event_rel"], "event_tfidf": eve_tfidf, "event_sentiment": eve_sent}
            neo4j.create_relation(rel_set=(head, rel, tail))
            # 實體提及關係
            for ent in eve_tri[0]["mentioned"]:
                rel = {"name": "涉及"}
                mention_ent = {"type": ent[1], "name": ent[0]}
                neo4j.create_relation(rel_set=(head, rel, mention_ent))
            for ent in eve_tri[2]["mentioned"]:
                rel = {"name": "涉及"}
                mention_ent = {"type": ent[1], "name": ent[0]}
                neo4j.create_relation(rel_set=(tail, rel, mention_ent))
    except Exception as e:
        print(e)
        print("\n")
        print(eve_tri)
        pass

    try:
        # 節點&關係創建: event_causality(event_causality_ent_mentions)
        for idx in range(len(_set["event_causality_ent_mentions"])):
            eve_causal = _set["event_causality_ent_mentions"][idx]
            eve_tfidf, eve_sent = _set["event_causality"][idx][1], _set["event_causality"][idx][2]
            head = {"type": "EVENT_REASON", "name": eve_causal[0]["reason"]}
            tail = {"type": "EVENT_RESULT", "name": eve_causal[1]["result"]}
            neo4j.create_node(node=head)
            neo4j.create_node(node=tail)
            rel = {"name": "涉及"}
            neo4j.create_relation(rel_set=(news, rel, head))
            rel = {"name": "導致", "event_tfidf": eve_tfidf, "event_sentiment": eve_sent}
            neo4j.create_relation(rel_set=(head, rel, tail))
            # 實體涉及關係
            for ent in eve_causal[0]["mentioned"]:
                rel = {"name": "涉及"}
                mention_ent = {"type": ent[1], "name": ent[0]}
                neo4j.create_relation(rel_set=(head, rel, mention_ent))
            for ent in eve_causal[1]["mentioned"]:
                rel = {"name": "涉及"}
                mention_ent = {"type": ent[1], "name": ent[0]}
                neo4j.create_relation(rel_set=(tail, rel, mention_ent))
    except Exception as e:
        print(e)
        print("\n")
        print(eve_causal)
        pass


# %%
