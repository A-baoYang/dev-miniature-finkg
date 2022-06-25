import json
import os
import pandas as pd

from KGBuilder.config import *


args['root_dir'] = os.getcwd()
input_dir = "/home/abao.yang@cathayholdings.com.tw/forResearch/data/gcs/Crawling/Financial/data/taiwan_financial_news/"
input_filename = "news-20220101_20220413-俄烏戰爭"
output_filename = os.path.join(args["article_dir"], f"fromCrawler_{input_filename}.csv")

# 載入爬蟲得到的新聞資料
with open(os.path.join(input_dir, input_filename) + ".json", "r", encoding="utf-8") as f:
    crawled_data = json.load(f)

columns = ["source", "title", "article", "summary", "keywords", "publish_date"]
df_data = pd.concat([
    pd.DataFrame(crawled_data[source], columns=columns) 
    for source in ["moneydj"] if source in crawled_data.keys() # "ltn","moneydj","technews"
]).drop(["summary"], axis=1)
df_data = df_data[df_data["publish_date"] >= "2022-01-01"].sort_values("publish_date", ascending=False).reset_index(drop=True)
print(df_data.shape)

df_data = df_data.rename(columns={"publish_date": "publish_time", "keywords": "meta_keywords"})
df_data["crawling_time"] = "2022-04-13 11:00:11"
df_data = df_data[["source","title","publish_time","article","meta_keywords","crawling_time"]]
print(df_data.head())
print(df_data.tail())

df_data.to_csv(output_filename, index=False)

