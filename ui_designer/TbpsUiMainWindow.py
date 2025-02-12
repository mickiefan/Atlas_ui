'''
Author: gengyou.lu 1770591868@qq.com
Date: 2025-01-07 10:34:13
FilePath: /Atlas200_tbps_ui/ui_designer/TbpsUiMainWindow.py
LastEditTime: 2025-01-11 11:24:27
Description: tbps ui main window
'''
import os
import sys
# 通过当前文件目录的相对路径设置工程的根目录
current_file_path = os.path.abspath(os.path.dirname(__file__))
project_base_path = os.path.abspath(os.path.join(current_file_path, "../"))
sys.path.append(project_base_path)

from deploy.deploy_tbps import tokenize, transfer_pic, net
from deploy.simple_tokenizer import SimpleTokenizer
from config import DEVICE_IS_ATLAS

import json
import numpy as np
from datetime import datetime
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget,QGraphicsScene
from .Ui_tbps import Ui_MainWindow 


class MyMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        # 设置执行平台
        self.is_atlas = DEVICE_IS_ATLAS
        # 初始化模型        
        if self.is_atlas:
            self.image_encoder = net(os.path.join(project_base_path, "deploy/model/xsmall_image_encode_310B4.om"))
            bpe_path = os.path.join(project_base_path, "data/bpe_simple_vocab_16e6.txt.gz")
            self.tokenizer = SimpleTokenizer(bpe_path)
            self.text_encoder = net(os.path.join(project_base_path, "deploy/model/xsmall_text_encode_310B4.om")) 
            self.consine_sim_model = net(os.path.join(project_base_path, "deploy/model/similarity_310B4.om")) 
        else:
            self.image_encoder = None
            self.tokenizer = None
            self.text_encoder = None
            self.consine_sim_model = None

        # 静态检索相关变量
        self.static_database_file_path = ""
        self.static_database_json_file_path = ""
        self.static_gt_image_path = ""
        self.dynamic_gt_image_path = ""
        self.current_search_gt_image_path = ""
        self.set_pid = "none"

        # 动态检索相关变量
        self.dynamic_database_base_path = "" # 所选数据集文件夹的父目录，用于构建图像完整路径
        self.dynamic_dataset_folder_name = "" # 所选数据集文件夹名称，用于构建存储图像特征文件名
        self.dynamic_database_image_files = []
        self.dynamic_image_features = None
        
        # 显示相关变量
        self.show_result_frame_list = [self.frame_show_result1, self.frame_show_result2, self.frame_show_result3, self.frame_show_result4, self.frame_show_result5, 
                                        self.frame_show_result6, self.frame_show_result7, self.frame_show_result8, self.frame_show_result9, self.frame_show_result10]
        self.show_images_label_list = [self.label_show_img1, self.label_show_img2, self.label_show_img3, self.label_show_img4, self.label_show_img5,
                                          self.label_show_img6, self.label_show_img7, self.label_show_img8, self.label_show_img9, self.label_show_img10]
        self.show_sim_label_list = [self.label_show_sim1, self.label_show_sim2, self.label_show_sim3, self.label_show_sim4, self.label_show_sim5,
                                    self.label_show_sim6, self.label_show_sim7, self.label_show_sim8, self.label_show_sim9, self.label_show_sim10]

        # 存储检索结果相关变量
        self.current_search_json_result = {}
        self.history_search_json_result = {}

        # 初始化 GT 显示
        search_style = self.comboBox_search_style.currentText() 
        self.show_gt_and_pid(search_style)


    def closeEvent(self, event):
        self.release_resources()
        event.accept()  # 接受关闭事件

    def release_resources(self): 
        # 释放模型资源
        if self.image_encoder is not None:
            del self.image_encoder
        if self.text_encoder is not None:
            del self.text_encoder
        if self.consine_sim_model is not None:
            del self.consine_sim_model

    # ************************ slot functions ************************ #
    def slot_select_dataset(self):        
        # 设置基础路径
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        project_base_path = os.path.abspath(os.path.join(current_file_path, "../data/static_database"))
        # print(project_base_path)
        # 打开文件选择对话框
        static_database_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', project_base_path)
        if static_database_file_path:
            self.lineEdit_select_dataset.setText(static_database_file_path)  # 设置选择的文件路径到 QLineEdit

    def slot_set_gt(self):
        # 静态检索的 GT 设置
        set_gt_flag = self.comboBox_GT.currentText()
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        if set_gt_flag == "无 GT":            
            self.lineEdit_GT.setText("")            
        elif set_gt_flag == "加载 GT":
            # 设置基础路径
            current_file_path = os.path.abspath(os.path.dirname(__file__))
            project_base_path = os.path.abspath(os.path.join(current_file_path, "../data/static_database"))            
            # 打开文件选择对话框
            gt_image_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', project_base_path)
            if gt_image_path:
                self.lineEdit_GT.setText(gt_image_path)  # 设置选择的文件路径到 QLineEdit
                self.static_gt_image_path = gt_image_path
        # 获取检索 style
        search_style = self.comboBox_search_style.currentText() 
        self.show_gt_and_pid(search_style)
        
    def slot_set_pid(self):
        # 静态检索的 PID 设置
        set_pid_flag = self.comboBox_PID.currentText()
        if set_pid_flag == "无 PID":
            self.lineEdit_PID.setText("") 
            self.set_pid = "none"
        elif set_pid_flag == "加载 PID":
            # 使用弹出窗口设置 PID
            text, ok = QtWidgets.QInputDialog.getText(self, '设置 PID', '请输入 PID:')
            if ok:
                self.lineEdit_PID.setText(text) 
                self.set_pid = text           

    def slot_select_path(self):
        # 设置基础路径
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        project_base_path = os.path.abspath(os.path.join(current_file_path, "../data/dynamic_database"))
        # print(project_base_path)
        # 打开文件选择对话框
        dynamic_database_path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择文件夹', project_base_path)
        if dynamic_database_path:
            self.lineEdit_select_path.setText(dynamic_database_path)

    def slot_set_search_style(self):
        search_style = self.comboBox_search_style.currentText()
        self.show_gt_and_pid(search_style)

    def slot_search(self):
        self.terminal_message("=========== Start Search ===========")
        # 获取输入文本描述
        enter_text_description = self.textEdit_enter_text_description.toPlainText()
        if enter_text_description == "":
            self.terminal_message("Please enter text description", is_error=True)
            return
        # 获取检索 style
        search_style = self.comboBox_search_style.currentText()        
        if search_style == "静态检索":
            self.terminal_message("Search style: Static Search")            
            self.terminal_message("Query:")
            self.terminal_message(enter_text_description)
            # 检查静态数据库及图像索引是否完备
            if self.check_static_database():
                # 清空结果显示，等待结果
                self.clean_show_result_before_search()
                # 静态检索
                result_sim, result_pids, result_image_paths, dataset_base_path = self.static_search(enter_text_description)                
                # 汇总检索结果
                self.catch_json_result(search_style, result_sim, result_pids, result_image_paths, dataset_base_path)
                # 展示 Top10 结果
                self.show_search_result(self.current_search_json_result)
                # 展示检索结果概要
                self.show_search_result_abstract(self.current_search_json_result)
            else:
                self.terminal_message("ERROR: Please check static database!", is_error=True)
                return
        elif search_style == "动态检索":
            self.terminal_message("Search style: Dynamic Search")            
            self.terminal_message("Query:")
            self.terminal_message(enter_text_description)            
            if self.get_dynamic_database():
                # 清空结果显示，等待结果
                self.clean_show_result_before_search()
                # 动态检索
                result_sim, result_image_ids, result_image_paths, dataset_base_path = self.dynamic_search(enter_text_description)                
                # 汇总检索结果
                self.catch_json_result(search_style, result_sim, result_image_ids, result_image_paths, dataset_base_path)
                # 展示 Top10 结果
                self.show_search_result(self.current_search_json_result)
                # 展示检索结果概要
                self.show_search_result_abstract(self.current_search_json_result)
            else:
                self.terminal_message("ERROR: Dynamic data path dose not contain an image file!", is_error=True)
                return
        else:
            self.terminal_message("Please select search style", is_error=True)
            return

    def slot_save_dataset(self):
        save_npy_name = self.lineEdit_dynamic_to_static_name.text()
        if save_npy_name.endswith('.npy') is False:
            self.terminal_message("Please enter a valid file name, such as '*.npy'.")
            return
        # 保存动态图像特征
        save_feature_path = os.path.join(self.dynamic_database_base_path, save_npy_name)        
        np.save(save_feature_path, self.dynamic_image_features)
        # 保存图像对应路径
        save_image_path = save_feature_path.replace('.npy', '.json')
        json_data = {"img_paths": self.dynamic_database_image_files}
        json.dump(json_data, open(save_image_path, 'w'), indent=4)
        self.terminal_message("Save dynamic dataset successfully!")

    def slot_clean_terminal_output(self):
        self.textBrowser_terminal_output.clear()

    def slot_save_result(self):
        if not self.current_search_json_result:
            self.terminal_message("Please search first!", is_error=True)
            return        
        # 设置基础路径
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        project_base_path = os.path.abspath(os.path.join(current_file_path, "../data/search_result"))        
        # 使用弹框设置保存路径
        # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = os.path.join(project_base_path, f"{current_time}.json")
                
        options = QtWidgets.QFileDialog.Options()
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, '保存文件', default_filename, 'JSON Files (*.json)', options=options)               
        if save_path:
            json.dump(self.current_search_json_result, open(save_path, 'w'), indent=4)
            self.terminal_message("Save search result successfully!")

    def slot_load_history_result(self):
        # 设置基础路径
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        project_base_path = os.path.abspath(os.path.join(current_file_path, "../data/search_result"))        
        # 打开文件选择对话框
        history_result_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', project_base_path, 'JSON Files (*.json)')
        if history_result_path:
            search_json_result = json.load(open(history_result_path, 'r'))
            # 检查结果是否完备
            if "search_style" not in search_json_result or \
                "query_text" not in search_json_result or \
                "similarity" not in search_json_result or \
                "result_imgids_or_pids" not in search_json_result or \
                "image_paths" not in search_json_result or \
                "dataset_base_path" not in search_json_result or \
                "gt_image_path" not in search_json_result or \
                "set_pid" not in search_json_result:

                self.terminal_message("ERROR: Please check history search result!", is_error=True)
                return            
            self.history_search_json_result = search_json_result
            self.terminal_message("Load history search result successfully!")
        # 展示历史检索结果
        self.show_search_result(self.history_search_json_result)
        self.show_search_result_abstract(self.history_search_json_result)

    def slot_show_current_result(self):
        if self.current_search_json_result == {}:
            self.terminal_message("Please search first!", is_error=True)
            return
        self.show_search_result(self.current_search_json_result)
        self.show_search_result_abstract(self.current_search_json_result)

    # ************************ deploy functions ************************ #
    def static_search(self, query_text):        
        # 1.读取静态数据库
        test_image_norm_features = np.load(self.static_database_file_path)
        N = test_image_norm_features.shape[0]        
        with open(self.static_database_json_file_path, 'r') as f:
            static_database_json = json.load(f)
        # 获取数据集 base 目录
        dataset_base_path = os.path.dirname(self.static_database_file_path)
        self.update_progress_bar(1, 5)
        if self.is_atlas:
            # 2.获取文本特征
            text = tokenize(query_text, tokenizer=self.tokenizer, text_length=77, truncate=True)
            text = text.reshape((1, 77))
            result = self.text_encoder.text_forward(text) # npu 计算     
            text_feature = result[text.argmax(axis=-1), :] # 获取最大值的索引对应的特征，即为文本的 cls 特征        
            self.update_progress_bar(2, 5)
            # 3.计算图像数据库特征与文本特征的相似度
            similarity, index = [], []
            loops = N // 1024
            for i in range(loops):
                # 准备图像数据
                start_index = i * 1024 
                end_index = min((i + 1) * 1024, N)
                images = test_image_norm_features[start_index:end_index]
                # DEBUG 文本数据
                # text_feature = images[0, :]
                # 准备start_index数据
                start_index = np.array([start_index], dtype=np.int64) 
                inputs = [images, text_feature, start_index]
                result = self.consine_sim_model.similarity_forward(inputs) # npu 计算  
                similarity.append(result[0])
                index.append(result[1])        
            # 处理不整除的情况
            if N % 1024 != 0:
                start_index = loops * 1024
                images = np.zeros((1024, 512), dtype=np.float32)
                images[0 : N - start_index] = test_image_norm_features[start_index:]
                start_index = np.array([start_index], dtype=np.int64)
                inputs = [images, text_feature, start_index]
                result = self.consine_sim_model.similarity_forward(inputs)
                similarity.append(result[0])
                index.append(result[1])
            self.update_progress_bar(3, 5)
            # 4.合并结果,并进行最终 TopK 操作    
            similarity = np.concatenate(similarity, axis=1)
            index = np.concatenate(index, axis=1)    
            # 获取前 K 个最大值的索引
            K = 10
            sorted_indices = np.argsort(similarity, axis=1)[:, ::-1]
            indices = sorted_indices[:, :K]
            top10_values = np.take_along_axis(similarity, indices, axis=1).flatten().tolist()
            top10_indices = np.take_along_axis(index, indices, axis=1).flatten().tolist()
            self.update_progress_bar(4, 5)
        else:
            # DEBUG for development on x86
            top10_values = np.random.rand(1, 10).flatten().tolist()
            # top10_indices = np.random.randint(0, N, (1, 10)).flatten() 
            top10_indices = np.array([1,1,1,2,3,4,5,6,7,8]).flatten().tolist()   
        # 5. 返回 Top10 的相似度值和对应的图像路径
        show_images_path =  [static_database_json['img_paths'][i] for i in top10_indices]
        result_image_ids = [static_database_json['image_pids'][i] for i in top10_indices]
        self.update_progress_bar(5, 5)
        return top10_values, result_image_ids, show_images_path, dataset_base_path

    def dynamic_search(self, query_text):
        # 设置数据集路径
        database_image_files = self.dynamic_database_image_files
        dataset_base_path = self.dynamic_database_base_path
        total_bar = len(database_image_files) + 10     
        if self.is_atlas:
            # 1.获取文本特征
            text = tokenize(query_text, tokenizer=self.tokenizer, text_length=77, truncate=True)
            text = text.reshape((1, 77))
            result = self.text_encoder.text_forward(text) # npu 计算     
            text_feature = result[text.argmax(axis=-1), :] # 获取最大值的索引对应的特征，即为文本的 cls 特征 
            self.update_progress_bar(5, total_bar)
            # 2.获取图像特征
            image_features = []
            i = 1
            for image_file in database_image_files:
                img_path = os.path.join(dataset_base_path, image_file)
                om_input_image = transfer_pic(img_path)
                result = self.image_encoder.image_forward(om_input_image)
                # 归一化 om 模型推理结果
                om_image_feat = result[0, :].reshape(1, -1)
                om_image_feat = om_image_feat / np.linalg.norm(om_image_feat, ord=2, axis=-1, keepdims=True)
                image_features.append(om_image_feat) 
                i = i + 1
                self.update_progress_bar(5 + i, total_bar)
            self.dynamic_image_features = np.concatenate(image_features, axis=0)                        
            N = self.dynamic_image_features.shape[0]        
            # 3.计算图像数据库特征与文本特征的相似度
            similarity, index = [], []
            loops = N // 1024
            for i in range(loops):
                # 准备图像数据
                start_index = i * 1024 
                end_index = min((i + 1) * 1024, N)
                images = self.dynamic_image_features[start_index:end_index]
                # DEBUG 文本数据
                # text_feature = images[0, :]
                # 准备start_index数据
                start_index = np.array([start_index], dtype=np.int64) 
                inputs = [images, text_feature, start_index]
                result = self.consine_sim_model.similarity_forward(inputs) # npu 计算  
                similarity.append(result[0])
                index.append(result[1])        
            # 处理不整除的情况
            if N % 1024 != 0:
                start_index = loops * 1024
                images = np.zeros((1024, 512), dtype=np.float32)
                images[0 : N - start_index] = self.dynamic_image_features[start_index:]
                start_index = np.array([start_index], dtype=np.int64)
                inputs = [images, text_feature, start_index]
                result = self.consine_sim_model.similarity_forward(inputs)
                similarity.append(result[0])
                index.append(result[1])
            self.update_progress_bar(total_bar - 3, total_bar)
            # 4.合并结果,并进行最终 TopK 操作    
            similarity = np.concatenate(similarity, axis=1)
            index = np.concatenate(index, axis=1)    
            # 获取前 K 个最大值的索引
            K = 10
            sorted_indices = np.argsort(similarity, axis=1)[:, ::-1]
            indices = sorted_indices[:, :K]
            top10_values = np.take_along_axis(similarity, indices, axis=1).flatten().tolist()
            top10_indices = np.take_along_axis(index, indices, axis=1).flatten().tolist()
        else:        
            # DEBUG for development on x86
            self.dynamic_image_features = np.random.randn(500, 512)
            N = self.dynamic_image_features.shape[0]
            top10_values = np.random.rand(1, 10).flatten().tolist()
            top10_indices = np.random.randint(0, N, (1, 10)).flatten().tolist()              
        # 5. 返回 Top10 的相似度值和对应的图像路径
        show_images_path =  [os.path.join(dataset_base_path, database_image_files[i]) for i in top10_indices]
        # 6. 设置保存动态图像特征文件名称
        self.lineEdit_dynamic_to_static_name.setText(f"{self.dynamic_dataset_folder_name}_test_data.npy")
        self.update_progress_bar(total_bar, total_bar)
        return top10_values, top10_indices, show_images_path, dataset_base_path

    # ************************ utils functions ************************ #
    def terminal_message(self, text, is_error=False):
        if is_error:
            self.textBrowser_terminal_output.append(f"<span style='color:red;'>{text}</span>")
        else:
            self.textBrowser_terminal_output.append(f"<span style='color:black;'>{text}</span>")
        self.textBrowser_terminal_output.moveCursor(self.textBrowser_terminal_output.textCursor().End)

    def check_static_database(self):
        static_database_file_path = self.lineEdit_select_dataset.text()
        if static_database_file_path is None:
            # 提示选择数据集
            self.terminal_message("Please select dataset", is_error=True)
            return False
        if static_database_file_path.lower().endswith('.npy') is False:
            # 提示选择.npy文件
            self.terminal_message("Please select '*.npy' file", is_error=True)
            return False         
        static_database_json_file_path = static_database_file_path.replace('.npy', '.json')
        if os.path.exists(static_database_json_file_path) is False:
            # 提示生成json文件
            self.terminal_message("Please generate json file for dataset", is_error=True)
            return False
        # 设置静态检索相关变量
        self.static_database_file_path = static_database_file_path
        self.static_database_json_file_path = static_database_json_file_path
        return True

    def get_dynamic_database(self):
        dynamic_database_path = self.lineEdit_select_path.text()
        if os.path.exists(dynamic_database_path) is False and os.path.isdir(dynamic_database_path) is False:
            # 提示选择数据集
            self.terminal_message("Please select true and exit data path", is_error=True)
            return False
        # 获取目录下的所有图像文件        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')        
        image_files = []
        basepath = os.path.basename(dynamic_database_path) # 提取最后一级目录，制作图像相对路径
        self.dynamic_dataset_folder_name = basepath
        for f in os.listdir(dynamic_database_path):
            if f.lower().endswith(image_extensions):
                image_files.append(os.path.join(basepath, f))
        # 检查是否有图像文件
        if len(image_files) != 0:
            # 设置动态检索相关变量
            # self.dynamic_database_base_path + self.dynamic_database_image_files[i] 可以获取图像完成路径
            self.dynamic_database_base_path = os.path.dirname(dynamic_database_path)
            self.dynamic_database_image_files = image_files
            return True
        return False

    def show_search_result(self, show_search_json_result={}):
        if show_search_json_result == {}:
            return
        # 获取检索结果
        search_style = show_search_json_result["search_style"]
        result_sim = show_search_json_result["similarity"]
        result_image_paths = show_search_json_result["image_paths"]
        dataset_base_path = show_search_json_result["dataset_base_path"]
        result_pids = show_search_json_result["result_imgids_or_pids"]
        pids = show_search_json_result["set_pid"]           
        # 展示 Top10 图像及相似度
        for i in range(10):
            image_path = os.path.join(dataset_base_path, result_image_paths[i])
            sim = result_sim[i] 
            pixmap = QPixmap(image_path)            
            resized_pixmap = pixmap.scaled(140, 180) 
            self.show_images_label_list[i].setPixmap(resized_pixmap)
            self.show_sim_label_list[i].setText(f"匹配度: {sim:.3f}")            
            self.show_images_label_list[i].setAlignment(QtCore.Qt.AlignCenter)
            self.show_sim_label_list[i].setAlignment(QtCore.Qt.AlignCenter)            
        # 设置 frame 边框显示
        if search_style == "静态检索":
            for i in range(10):                               
                if str(result_pids[i]) == pids:
                    self.show_frame_border(i, show_border=True) 
                else:
                    self.show_frame_border(i, show_border=False)                                   
        else:
            for i in range(10):
                self.show_frame_border(i, show_border=False) 

    def update_progress_bar(self, i, N):
        value = int(i / N * 100)
        # 更新进度条
        self.progressBar.setValue(value)
    
    def show_gt_and_pid(self, search_style):
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        
        if search_style == "静态检索":
            # GT 设置 
            set_gt_flag = self.comboBox_GT.currentText()
            if set_gt_flag == "无 GT":                
                self.static_gt_image_path = os.path.join(current_file_path, "ui_data/ui_no_gt.jpg")                                              
            elif set_gt_flag == "加载 GT":
                gt_image_path = self.lineEdit_GT.text()
                if os.path.exists(gt_image_path) is False:
                    # 提示选择数据集
                    self.terminal_message("Please select true gt data path", is_error=True)
                    return False
                # 获取目录下的所有图像文件        
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')        
                if gt_image_path.lower().endswith(image_extensions):
                    self.static_gt_image_path = gt_image_path                    
                else:
                    self.terminal_message("Please select image file for setting GT", is_error=True)
                    return False 
            self.current_search_gt_image_path = self.static_gt_image_path                        
        elif search_style == "动态检索":
            # 显示无 GT 图像            
            self.dynamic_gt_image_path = os.path.join(current_file_path, "ui_data/ui_no_gt.jpg")
            self.current_search_gt_image_path = self.dynamic_gt_image_path
        # 显示 gt
        pixmap = QPixmap(self.current_search_gt_image_path)
        resized_pixmap = pixmap.scaled(90, 120)                     
        self.label_show_gt.setPixmap(resized_pixmap) 
        self.label_show_gt.setScaledContents(True)                                   
        self.label_show_gt.setAlignment(QtCore.Qt.AlignCenter) 

    def catch_json_result(self, search_style, result_sim, result_imgids_or_pids, result_image_paths, dataset_base_path):
        self.current_search_json_result["search_style"] = str(search_style)
        self.current_search_json_result["query_text"] = self.textEdit_enter_text_description.toPlainText()        
        self.current_search_json_result["similarity"] = result_sim
        self.current_search_json_result["result_imgids_or_pids"] = result_imgids_or_pids
        self.current_search_json_result["image_paths"] = result_image_paths
        self.current_search_json_result["dataset_base_path"] = dataset_base_path
        self.current_search_json_result["gt_image_path"] = self.current_search_gt_image_path
        self.current_search_json_result["set_pid"] = self.set_pid

    def show_search_result_abstract(self, show_search_json_result={}):        
        
        if show_search_json_result == {}:
            return
        
        # 设置背景颜色                
        self.frame_query_abstract.setStyleSheet("background-color: rgb(215, 227, 243);")
        # 设置标题
        self.label_query_abstract.setText("检索概要总览")
        # 展示 GT 图像
        gt_image_path = show_search_json_result["gt_image_path"]
        pixmap = QPixmap(gt_image_path)
        resized_pixmap = pixmap.scaled(90, 120)             
        self.label_show_result_gt_image.setPixmap(resized_pixmap)
        self.label_show_result_gt_image.setScaledContents(True)                            
        self.label_show_result_gt_image.setAlignment(QtCore.Qt.AlignCenter) 
        # 居中展示 GT 标签            
        self.label_show_result_gt_label.setText("GT")            
        self.label_show_result_gt_label.setAlignment(QtCore.Qt.AlignCenter)
        # 居中展示 PID 标签  
        pids = show_search_json_result["set_pid"]          
        self.label_show_result_PID_label.setText("PID: " + pids)            
        self.label_show_result_PID_label.setAlignment(QtCore.Qt.AlignCenter)
        # 展示 Top1 结果
        top1_image_path = os.path.join(show_search_json_result["dataset_base_path"], 
                                        show_search_json_result["image_paths"][0])
        pixmap = QPixmap(top1_image_path)
        resized_pixmap = pixmap.scaled(90, 120)
        self.label_show_result_top1_image.setPixmap(resized_pixmap)
        self.label_show_result_top1_image.setAlignment(QtCore.Qt.AlignCenter)
        # 居中展示 Top1 标签            
        self.label_show_result_top1_label.setText("Top 1")            
        self.label_show_result_top1_label.setAlignment(QtCore.Qt.AlignCenter)
        # 居中展示匹配度
        top1_sim = show_search_json_result["similarity"][0]
        self.label_show_result_sim_label.setText(f"匹配度: {top1_sim:.3f}")
        self.label_show_result_sim_label.setAlignment(QtCore.Qt.AlignCenter)
        # 展示检索信息总览
        self.textBrowser_show_result_abstract.clear()
        self.textBrowser_show_result_abstract.append(f"检索方式: {show_search_json_result['search_style']}")
        self.textBrowser_show_result_abstract.append(f"查询文本: {show_search_json_result['query_text']}")

    def show_frame_border(self, i, show_border=False):        
        if show_border:
            border_style = "2px solid rgb(255, 0, 0)"
            self.show_result_frame_list[i].setStyleSheet(f"""
                QFrame {{
                    border: {border_style};
                    background-color: transparent;
                }}
            """)
            self.show_images_label_list[i].setStyleSheet("""
                QFrame {
                    border: none;
                    background-color: transparent;
                }
            """)
            self.show_sim_label_list[i].setStyleSheet("""
                QFrame {
                    border: none;
                    background-color: transparent;
                }
            """)
        else:
            self.show_result_frame_list[i].setStyleSheet("""
                QFrame {
                    border: none;
                    background-color: transparent;
                }
            """)

    def clean_show_result_before_search(self):
        # 清空 Top10 结果
        for i in range(10):
            self.show_images_label_list[i].clear()
            self.show_sim_label_list[i].clear()              
            self.show_frame_border(i, show_border=False)
                                
        # 清空检索概要总览  
        self.frame_query_abstract.setStyleSheet("background-color: rgb(226, 239, 255);")      
        self.label_query_abstract.setText("")                          
        self.label_show_result_gt_image.clear()
        self.label_show_result_gt_label.setText("")                            
        self.label_show_result_PID_label.setText("")                    
        self.label_show_result_top1_image.clear()        
        self.label_show_result_top1_label.setText("")                                    
        self.label_show_result_sim_label.setText("")                
        self.textBrowser_show_result_abstract.clear()

