import json
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from KGBuilder.config import *


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def file_io(method, filepath, data=None, newline_json=False):
    if method == "read":
        if ".json" in filepath:
            import json 
            if not newline_json:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = []
                    file_str = f.read()
                    for item in tqdm(file_str.split("\n")):
                        try:
                            data.append(eval(item))
                        except Exception as e:
                            print(e, item)
                            pass

        elif ".csv" in filepath:
            import pandas as pd
            data = pd.read_csv(filepath)
        elif ".txt" in filepath:
            with open(filepath, "r", encoding="utf-8") as f:
                data = f.read()
        return data
    elif method == "write":
        if ".json" in filepath:
            import json 
            if not newline_json:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            else:
                with open(filepath, "a+", encoding="utf-8") as f:
                    for j in tqdm(data):
                        f.write(f"{json.dumps(j, cls=NpEncoder)}\n")
        elif ".csv" in filepath:
            import pandas as pd
            data.to_csv(filepath, index=False)
        elif ".txt" in filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                for j in tqdm(data):
                    f.write(j)
    


if __name__ == "__main__":
    args["root_dir"] = os.getcwd()

    # FinBERT
    # 詞彙層級的情緒分數
    from KGBuilder.OpenModels.sentiments import SentimentAnalyzer

    sent_analyzer = SentimentAnalyzer(module_name="FinBERT")
    sentiment_results = sent_analyzer.finbert_predict(input_filepath="cleansed-quick-technews-俄烏戰爭.csv")

    article_filepath = "article_data/cleansed-quick-technews-俄烏戰爭.csv"
    finer_filepath = "model_data/output/merged_ner_results-%s.json" % args["timestamp"]
    sent_filepath = "model_data/output/FinKG-finbertSA-%s.json" % args["timestamp"]
    nersent_filepath = "model_data/output/ner_sent_merged-%s.json" % args["timestamp"]
    mergedAll_filepath = "model_data/output/merged_all-%s.json" % args["timestamp"]

    article_data = file_io(method="read", filepath=article_filepath)
    finer_data = file_io(method="read", filepath=finer_filepath)
    sent_data = file_io(method="read", filepath=sent_filepath)
    print(len(article_data), len(finer_data), len(sent_data))

    # (quick) merge FinBERT to FiNER results
    merge_finer_data = finer_data.copy()
    for i in tqdm(range(len(merge_finer_data))):
        splited_text_and_sent = [(item["text"], item["sentiment_score"]) for item in sent_data[i]["splited_text"]]
        merge_finer_data[i].update({"entity_sentiment": {}})
        for k in merge_finer_data[i]["entity_text"]:
            _item_sent = []
            for item in merge_finer_data[i]["entity_text"][k]:
                sum_sent = 0
                sum_mentioned = 0
                for _set in splited_text_and_sent:
                    if item in _set[0]:
                        sum_sent += _set[1]
                        sum_mentioned += 1
                if sum_mentioned:
                    _item_sent.append((item, round(sum_sent / sum_mentioned, 3)))
                else:
                    _item_sent.append((item, 0.0))
            merge_finer_data[i]["entity_sentiment"].update({k: _item_sent})

    file_io(method="write", filepath=nersent_filepath, data=merge_finer_data)

    # (quick) Merge All 
    ner_sent_data = file_io(method="read", filepath=nersent_filepath)
    merge_all = []
    for i in tqdm(range(len(article_data))):
        _results = {
            "media": article_data["source"][i],
            "title": article_data["title"][i],
            "url": article_data["url"][i],
            "article": article_data["article"][i],
            "summary": article_data["description"][i],
            "publish_time": article_data["publish_time"][i],
            "article_sentiment": sent_data[i]["all_text"]["sentiment_score"],
            "entities": ner_sent_data[i]["entity_text"],
            "entities_sentiment": ner_sent_data[i]["entity_sentiment"],
        }
        merge_all.append(_results)
    print(merge_all[0])
    file_io(method="write", filepath=mergedAll_filepath, data=merge_all, newline_json=False)

    # Read the newline json file for check
    merge_all_data = file_io(method="read", filepath=mergedAll_filepath, newline_json=False)
    print(merge_all_data[0])