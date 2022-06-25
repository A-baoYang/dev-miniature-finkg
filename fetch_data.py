from glob import glob
import os
import pandas as pd
from tqdm import tqdm
from KGBuilder.config import *
from KGBuilder import octoparse
from KGBuilder.data_utils import colclean_time


print("Fetching data from %s to %s" % (args["start_date"], args["end_date"]))

# Octoparse API 身份驗證
base_url, username, password = octoparse.loginInfo(is_china=False)
print(base_url, username, password)
api_headers = octoparse.getAccessToken(base_url, username, password)
print(api_headers)

# 取得 Octoparse tasks from "MyGroup"
group_dict = octoparse.getTaskGroup(base_url, api_headers)
group_dict = {g["taskGroupName"]: g["taskGroupId"] for g in group_dict}
tasks = octoparse.getTaskId(
    base_url, api_headers, taskGroupId=str(group_dict["MyGroup"])
)

# 從 Octoparse Crawler 資料庫取得指定筆數的文章資料，並存到 article_data/ 下
for task in tqdm(tasks):
    export_data = octoparse.runFetchTask(
        base_url, api_headers, task, 
        offset_history=0, size=500, fetch_len="all"
    )
    print(export_data.shape, '\n', export_data.head())

# 為了後續應用方便，合併多個網域的新聞成一份
df = pd.DataFrame([])
path_pattern = os.path.join(
    args["root_dir"], args["article_dir"], "*-%s.csv" % (args["timestamp"])
)
output_filepath = os.path.join(
    args["root_dir"], args["article_dir"], data_args["article_filename"]
)
for path in tqdm(glob(path_pattern)):
    _df = pd.read_csv(path)
    df = pd.concat([df, _df])
    os.remove(path)

df = colclean_time(data=df, time_col="publish_time")
df["publish_time"] = df["publish_time"].astype(str)
df = df[
    (df["article"].notnull()) & 
    (df["title"].notnull()) & 
    (df["publish_time"] >= args["start_date"]) & 
    (df["publish_time"] < args["end_date"])
].reset_index(drop=True)
df.to_csv(output_filepath, index=False)
