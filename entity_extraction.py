from collections import Counter, defaultdict
from fuzzychinese import FuzzyChineseMatch
import gc
import json
import logging
logging.getLogger("fuzzychinese").setLevel(logging.CRITICAL)
from nltk.util import ngrams
import os
import pandas as pd
import shutil
from time import time
from tqdm import tqdm
from KGBuilder.config import *
from KGBuilder.EntityMerge.entity_utils import merge2dicts


def check_char_match_ratio(query, target, num_gram=2):
    correct, total = 0, 0
    match_word = []
    query = ["".join(list(k)) for k in dict(Counter(ngrams(query, num_gram))).keys()]
    target = ["".join(list(k)) for k in dict(Counter(ngrams(target, num_gram))).keys()]
    for w in query:
        if w in target:
            correct += 1
            match_word.append(w)
        total += 1
    res = round(correct / total, 4) if total else 0.0
    return [res, match_word]


def do_fuzzymatch(query):
    # FuzzyChineseMatch intialize
    fcm = FuzzyChineseMatch(ngram_range=(3, 3), analyzer='stroke')
    fcm.fit(list(stock_ids.keys()))
    top_similar = fcm.transform(query, n=1)
    fuzzymatched = pd.DataFrame({
        "query": query,
        "top1_sim": top_similar.reshape(-1),
        "top1_score": fcm.get_similarity_score().reshape(-1),
    })
    fuzzymatched["2gram_match"] = fuzzymatched.apply(lambda row: check_char_match_ratio(query=row["query"], target=row["top1_sim"]), axis=1)
    t = fuzzymatched["2gram_match"].apply(pd.Series)
    t.columns = ["2gram_match_ratio","2gram_match_word"]
    fuzzymatched = pd.concat([fuzzymatched, t], axis=1).drop(["2gram_match"], axis=1)
    return fuzzymatched


if __name__ == "__main__":
    
    start = time()
    # args["timestamp"] = 20220807
    input_filepath = os.path.join(
        args["root_dir"], args["model_dir"], "input", # "event-triplets-input-%s.json" % args["timestamp"]
        data_args["module_EventTri_input_filename"]
    )
    stockid_filepath = os.path.join(
        args["root_dir"], gcs_args["gcs_dict_directory"],
        "twse_securities_ids.csv"
    )
    output_filepath = os.path.join(
        args["root_dir"], args["model_dir"], "output", # "entity-extraction-output-%s.json" % args["timestamp"]
        data_args["module_FiNER_output_filename"]
    )
    # 將 FiNER 所需的 slot_label.txt 從 dictionary 資料夾移過來
    filepath = os.path.join(args["root_dir"], args["model_dir"], "input", "slot_label.txt")
    if not os.path.exists(filepath):
        shutil.copyfile(
            os.path.join(args["root_dir"], gcs_args["gcs_dict_directory"], "slot_label.txt"), 
            filepath
        )
    # import twse stock list
    stock_ids = pd.read_csv(stockid_filepath)
    stock_ids = {item["name"]: item["id"] for item in stock_ids[["id","name"]].to_dict("records")}
    print("Num Stock: ", len(stock_ids))

    # CKIP Tagger || CKIP Transformers
    from KGBuilder.ckipNER.ckip import CKIPWrapper
    ckip = CKIPWrapper()
    input_data = ckip.data_loader(input_filepath=input_filepath)
    # article
    input_text_list = [input_data[i]["article"] for i in tqdm(range(len(input_data)))]
    _res_ckip_list = [_res_ckip["entity_text"] for _res_ckip in ckip.ner_predict(input_text=input_text_list)]

    # FiNER
    from KGBuilder.FiNER.trainer import FiNER
    from KGBuilder.FiNER.utils import (
        load_tokenizer,
        set_seed,
    )

    # 隨機亂數 & 模型參數初始化 / 載入 bert tokenizer & FiNER model
    set_seed()
    tokenizer = load_tokenizer()
    finer = FiNER()
    finer.load_model()
    stock_fuzzy_threshold = 0.8
    # 開始預測實體種類
    # article
    def finer_res_clean(res):
        res["EVENT"] = res.pop("EVE")
        res["PRODUCT"] = res.pop("GOO")
        return res

    text_tokens_list = [" ".join([word for word in text]) for text in tqdm(input_text_list)]
    _res_finer_list = [finer_res_clean(_res_finer["entity_text"]) for _res_finer in finer.predict(text_tokens_list, tokenizer)]
    # merge 2 dicts
    _res_list = [merge2dicts(dict1=_res_ckip_list[i], dict2=_res_finer_list[i], method="append") for i in tqdm(range(len(_res_finer_list)))]
    _ent_org = {i: _res["ORG"] for i, _res in enumerate(_res_list)}
    # arrange all ORG entities
    all_ent_org = []
    for l in _ent_org.values():
        all_ent_org += list(l)
    all_ent_org = list(set(all_ent_org))
    # ORG entity fuzzymatch as STOCK
    fuzzymatched = do_fuzzymatch(query=all_ent_org)
    matched_dict = fuzzymatched[(fuzzymatched["2gram_match_ratio"]>0) & (fuzzymatched["top1_score"]>stock_fuzzy_threshold)][["query","top1_sim"]].to_dict("records")
    matched_dict = {item["query"]: item["top1_sim"] for item in matched_dict}
    matched_dict = defaultdict(str, matched_dict)
    _ent_stock = [[stock_ids[matched_dict[name]] for name in l if matched_dict[name]] for l in tqdm(_ent_org.values())]

    # enhance: suitable format for search engine
    _res_list = [[[{"entity": name, "entity_type": type} for name in names] for type, names in _res.items()] for _res in tqdm(_res_list)]
    for i, _res in enumerate(tqdm(_res_list)):
        _tmp = []
        for item in _res:
            _tmp += item
        _res_list[i] = _tmp
        _res_list[i] += [{"entity": _stock, "entity_type": "STOCK"} for _stock in _ent_stock[i]]

    for i in tqdm(range(len(input_data))):
        input_data[i]["entity_extract"] = _res_list[i]


    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(input_data, f, ensure_ascii=False, indent=4)

    gc.collect()

    print("Duration: ", time() - start)
