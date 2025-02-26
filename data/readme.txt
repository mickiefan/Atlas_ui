# 数据集存储目录说明

(1) dynamic_database
# 存储静态检索数据库目录
# 静态检索数据库组织形式
|-- *_test_dataset
|   |-- cuhk-pedes // 存储图像目录
|   |-- cuhk-pedes_test_data.json // TODO： 待存储文件。存储图像相对路径，从 cuhk-pedes 开始
|   |-- cuhk-pedes_test_data.npy // TODO： 待存储文件。存储图像特征，经过图像编码器编码归一化的特征集    

* 注： 文件名 cuhk-pedes_test_data 由用户在交互界面中输入


(2) static_database
# 存储静态检索数据库目录
# 静态检索数据库组织形式
|-- *_test_dataset
|   |-- cuhk-pedes // 存储图像目录
|   |-- cuhk-pedes_test_data.json // 存储图像相对路径，从 cuhk-pedes 开始
|   |-- cuhk-pedes_test_data.npy // 存储图像特征，经过图像编码器编码归一化的特征集    
