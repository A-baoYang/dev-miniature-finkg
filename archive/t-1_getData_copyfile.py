import os
import pandas as pd
import shutil

from KGBuilder.config import *


args['root_dir'] = os.getcwd()

source_dir = "/home/abao.yang@cathayholdings.com.tw/forResearch/data/gcs/Crawling/Financial/data/taiwan_financial_news/"
# filename = "quick-technews-俄烏戰爭.csv"
filename = "test_reports.csv"
shutil.copy(
    os.path.join(source_dir, filename), 
    os.path.join(args["root_dir"], args["article_dir"], filename)
)

data = pd.read_csv(os.path.join(args["root_dir"], args["article_dir"], filename))
data = data.rename(columns={"keywords": "meta_keywords"})
data["source"] = "technews"
data["publish_time"] = data["url"].apply(lambda x: "-".join(x.split("/")[3:6]))

data.to_csv(
    os.path.join(args["root_dir"], args["article_dir"], filename), 
    index=False
)