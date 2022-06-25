import os
from KGBuilder.config import *
from KGBuilder import data_utils


# 下載所需字典檔(公司列表, FiNER slot label, FinRE relation schema)
source_dir = gcs_args["gcs_dict_directory"]
download_to_dir = os.path.join(args["root_dir"], gcs_args["gcs_dict_directory"])
data_utils.gcs_downloader(source_dir, download_to_dir)

# 載入公司列表(舊版比對公司名稱用)
company_list = data_utils.load_companies()
print(len(company_list))

# 載入文章資料集
article_data = data_utils.load_article(article_filepath=data_args["article_filename"])

# 進行文章前處理
preprocessed_data = data_utils.do_preprocess(data=article_data)
print(preprocessed_data.shape)
print(preprocessed_data.head())
