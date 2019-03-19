# ProjectName : SVMForPointNet
# FileName : remove_duplication
# Created on : 13/03/20199:48 AM
# Created by : KwanHoon Lee

import pandas as pd

config = {
    'filename':'duplex_all_yonsei.csv',
    'path':'./result/',
    'result_path' : './result/drop_duplications/'
}

data = pd.read_csv(config['path'] + config['filename'], header=0, index_col=0)
data.index = range(0, len(data))
data = data.round(4)
find_duplicate = data[['X', 'Y', 'Z', 'areas', 'ax1s', 'ax2s', 'gyrations', 'volumes']]
found_data = find_duplicate.drop_duplicates()

result = data.loc[found_data.index]
result.index = range(0, len(result))
result.to_csv(config['result_path'] + config['filename'])