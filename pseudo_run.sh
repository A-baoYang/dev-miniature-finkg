#!/bin/bash
# 從 Octoparse 抓取資料
python fetch_data.py
# 文本前處理、清洗
python preprocess.py
# 轉換成符合模型輸入的格式
python transform.py --module="EventTri"
# 事件三元組及因果關係抽取
python event_triplet.py
# 情緒分數計算 (to-be-update)
# python sentiment.py
# 實體抽取與事件所含實體比對
python entity_extraction.py
# 事件篩選及事件嵌套關係
python event_opt.py
# 產出相似實體表
python find_similar.py
# 合併相似實體 (to-be-update)
# python entity_merge.py
# 儲存節點、關係至 Neo4J 資料庫
python neo4j_store.py