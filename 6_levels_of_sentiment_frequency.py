#%%
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from KGBuilder.config import *
from sentiment_ana_n_merge import file_io
from KGBuilder.OpenModels.ckip import CKIPWrapper

args["root_dir"] = os.getcwd()
input_filepath = "model_data/output/merged_all-%s.json" % args["timestamp"]
stopwords_zhtw_filepath = "dictionary/stopwords_zhTW.txt"

#%%
# 載入資料
merge_all_data = file_io(method="read", filepath=input_filepath, newline_json=False)
stopwords_zhtw = file_io(method="read", filepath=stopwords_zhtw_filepath)
stopwords_zhtw = stopwords_zhtw.split("\n")

# %%
# 基於統計的重要度（盡量考慮獨特性和頻次）
# 實體詞層級: TF-IDF score
vectorizer = TfidfVectorizer()
ckip = CKIPWrapper()  # 初始化約需 40 sec

# %%
# ckip 基礎分詞
word_list = ckip.ckiptagger_ws([merge_all_data[i]["article"] for i in range(len(merge_all_data))])
# 排除停用詞
removed_stopword_wordlist = []
for l in word_list:
    _l = []
    for w in l:
        if w not in stopwords_zhtw:
            _l.append(w)
    removed_stopword_wordlist.append(_l)
word_list = removed_stopword_wordlist.copy()
del removed_stopword_wordlist, l, _l, w
# 生成每句話的 TF-IDF vectors, 約需 1 min
ws_tfidf_vector = vectorizer.fit_transform([' '.join(l) for l in word_list])
print(vectorizer.get_feature_names()[:10])
print(vectorizer.get_feature_names()[-10:])
print(ws_tfidf_vector.shape)

# %%
# 從每句話的 TF-IDF vectors mapping 該詞的重要程度 (TfidfVectorizer 已用今日所有文章為pool計算)
for i in tqdm(range(len(merge_all_data))):
    tfidf_dict = dict(zip(vectorizer.get_feature_names(), ws_tfidf_vector[i].toarray().reshape(-1)))
    tfidf_dict = {k: v for k, v in tfidf_dict.items() if v > 0}
    data = merge_all_data[i]
    _dict = {}
    # 實體層級: mapping 相對應的詞彙的TF-IDF分數，條件為包含或完全符合
    for type in data["entities"]:
        entity_tfidf = []
        for ent in data["entities"][type]:
            ent_tfidf = 0.0
            for word in tfidf_dict:
                if word in ent:
                    ent_tfidf += tfidf_dict[word]
            entity_tfidf.append((ent, round(ent_tfidf / len(ent) * 100, 3)))
        _dict.update({type: entity_tfidf})
    data.update({"entities_tfidf": _dict})

    for col in ["event_triplets", "event_causality"]:
        temp_event_triplets = []
        for event in data[col]:
            # 1) 事件層級: 詞TF-iDF分數的總和除以事件句子字數
            event_tfidf, event_sent = 0.0, 0.0
            for word in tfidf_dict:
                if word in event:
                    event_tfidf += tfidf_dict[word]

            # 2) 針對事件句子的情緒分數，用提及實體詞的情緒分數和實體詞層級重要度分數進行加權平均
            ent_sent_set, ent_tfidf_set = [], []
            event_sent, total_tfidf = 0.0, 0.0
            for type in data["entities_sentiment"]:
                for _set in data["entities_sentiment"][type]:
                    ent_sent_set.append(_set)
            for type in data["entities_tfidf"]:
                for _set in data["entities_tfidf"][type]:
                    ent_tfidf_set.append(_set)
            assert len(ent_sent_set) == len(ent_tfidf_set)
            
            for _id in range(len(ent_sent_set)):
                if ent_sent_set[_id][0] in event:
                    event_sent += ent_sent_set[_id][1] * ent_tfidf_set[_id][1]
                    total_tfidf += ent_tfidf_set[_id][1]
            temp_event_triplets.append((
                event, 
                round(event_tfidf / len(event) * 100, 3), 
                round(event_sent / total_tfidf, 3) if total_tfidf > 0 else 0.0
            ))
        data.update({col: temp_event_triplets})
    
    # 文章層級: 詞TF-iDF分數的總和除以文章字數
    # 每幾字出現較重要詞彙（冗詞少的文章排序高）
    data.update({
        "article_tfidf": round(sum(tfidf_dict.values()) / len(data["article"]) * 100, 3)
    })


# %%
output_filepath = "model_data/output/merged_all-%s-step6.json" % args["timestamp"]
file_io(method="write", filepath=output_filepath, data=merge_all_data)

# %%
# 事件情緒 & 重要度 table
df = pd.DataFrame({})
for data in tqdm(merge_all_data):
    id = merge_all_data.index(data)
    title = data["title"]
    event_types = ["triplet"] * len(data["event_triplets"]) + ["causality"] * len(data["event_causality"])
    event_sets = data["event_triplets"] + data["event_causality"]

    df = pd.concat([
        df, 
        pd.concat([
            pd.DataFrame({
                "Id": id,
                "Title": title,
                "Event Types": event_types
            }), 
            pd.DataFrame(event_sets, columns=["Events","Sentence-based TFIDF","Sentence-based FinBERT Sentiment Score"])
        ], axis=1)
    ])

df = df.reset_index(drop=True)
df.to_csv("model_data/output/temp_event_analysis_table.csv", encoding="utf-8", index=False)

# %%
# 事件濃縮(insight<=濃縮資訊)
# 將實體關係詞進行分類、將事件內容分類


