import os
from KGBuilder.config import *
from KGBuilder import data_utils

print(args)

args["root_dir"] = os.getcwd()
gcs_args[
    "service_account_key"
] = "../../forResearch/data/gcs/dst-dev2021-688422080376.json"
data_utils_args["input_filename"] = "quick-technews-俄烏戰爭.csv"
data_utils_args["cleansed_filename"] = "cleansed-%s" % (data_utils_args["input_filename"])


# 下載所需字典檔(公司列表, FiNER slot label, FinRE relation schema)
source_dir = gcs_args["gcs_dict_directory"]
download_to_dir = gcs_args["gcs_dict_directory"]
data_utils.gcs_downloader(source_dir, download_to_dir)

# 載入公司列表(舊版比對公司名稱用)
company_list = data_utils.load_companies()
print(len(company_list))

# 載入文章資料集
article_data = data_utils.load_article(article_filepath=data_utils_args["input_filename"])

# 進行文章前處理
media = args["media"]
expanded_df = data_utils.do_preprocess(
    data=article_data, media=media
)
print(expanded_df.shape)
print(expanded_df.head())

# 將處理後文章轉換為模型訓練用格式
df = data_utils.input_formatting(df=expanded_df)
print(df.shape)
print(df.head())
