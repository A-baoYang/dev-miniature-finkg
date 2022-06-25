import os
from tqdm import tqdm
from KGBuilder.config import *
from KGBuilder import neo4j


print("====== Load Data ======")
filepath = os.path.join(
    args["root_dir"], args["model_dir"], "output", 
    data_args["module_FiNER_output_filename"]
)
with open(filepath, "r", encoding="utf-8") as f:
    node_rels = json.load(f)

print(f"Length of company relations: {len(node_rels)}")
print(node_rels[0])

# 連線到 NEO4J 資料庫
connection = neo4j.DataToNeo4j(
    host=account_args["neo4j_hostname"],
    username=account_args["neo4j_username"],
    password=account_args["neo4j_password"],
    database=account_args["neo4j_database"],
)

for item in tqdm(node_rels):
    # NEWS node
    article_node = item.copy()
    event_triplets = article_node.pop("event_triplets")
    entity_extract = article_node.pop("entity_extract")
    article_node.update({"type": "NEWS"})
    article_node.update({"name": article_node["title"]})
    connection.create_node(node=article_node)

    for ent_type in tqdm(entity_extract.keys()):
        entities = entity_extract[ent_type]
        for name in entities:
            # 各類型實體 node
            ent_node = {"type": ent_type, "name": name}
            connection.create_node(node=ent_node)

    for event_item in tqdm(event_triplets):
        event_text = event_item["event"]
        # EVENT_HEAD, EVENT_TAIL node
        event_head = {"type": "EVENT_HEAD", "name": event_text[0]}
        event_tail = {"type": "EVENT_TAIL", "name": event_text[2]}
        connection.create_node(node=event_head)
        connection.create_node(node=event_tail)
        # EVENT_HEAD -事件關係-> EVENT_TAIL
        event_rel = {"type": event_text[1]}
        connection.create_relation(
            ent1=event_head, ent2=event_tail, rel=event_rel
        )
        # EVENT_HEAD/EVENT_TAIL -涉及-> (entities)
        event_ent_extract = event_item["entity_extract"]
        _rel = {"type": "涉及"}
        for _eve in [event_head, event_tail]:
            for _type, _ent_list in event_ent_extract.items():
                ent_mentioned = [{"type": _type, "name": ent} for ent in _ent_list if ent in _eve["name"]]
                if ent_mentioned:
                    for _node in tqdm(ent_mentioned):
                        connection.create_relation(
                            ent1=_eve, ent2=_node, rel=_rel
                        )

            # NEWS -涉及-> EVENT_HEAD
            connection.create_relation(
                ent1=article_node, ent2=_eve, rel=_rel
            )

print("Nodes & relations stored to Neo4j successfully.")
