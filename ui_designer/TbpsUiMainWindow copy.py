'''
Author: gengyou.lu 1770591868@qq.com
Date: 2025-01-07 10:34:13
FilePath: /Atlas200_tbps_ui/ui_designer/TbpsUiMainWindow.py
LastEditTime: 2025-01-11 11:24:27
Description: tbps ui main window
'''
import os
os.environ["QT_GSTREAMER_PLAYBIN_AUDIOSRC"] = "autoaudiosrc"
os.environ["QT_GSTREAMER_PLAYBIN_VIDEOSRC"] = "autovideosrc"
os.environ["QT_GSTREAMER_PLAYBIN"] = "ffmpeg"
import shutil
import sys
import numpy as np
import cv2 as cv
# 通过当前文件目录的相对路径设置工程的根目录
current_file_path = os.path.abspath(os.path.dirname(__file__))
project_base_path = os.path.abspath(os.path.join(current_file_path, "../"))
sys.path.append(project_base_path)

from deploy.deploy_tbps import tokenize, transfer_pic, net
from deploy.simple_tokenizer import SimpleTokenizer
from deploy.deploy_detection import detection_net
from deploy.progress import post_process, deal_result, yuv420sp_to_rgb
from config import DEVICE_IS_ATLAS

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow,QFileDialog, QApplication
from .Ui_tbps import Ui_MainWindow 
from acllite.acllite_model import AclLiteModel

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
            self.detection_model = AclLiteModel(os.path.join(project_base_path, "deploy/model/yolov4_bs1.om"))  # 行人检测模型
            self.detection_net = detection_net()
        else:
            self.image_encoder = None
            self.tokenizer = None
            self.text_encoder = None
            self.consine_sim_model = None
            self.detection_model = None
            self.detection_net = None

        # GT显示相关变量
        self.gt_image_path = ""

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
        self.show_result_frame_list = [self.frame_show_result1, self.frame_show_result2, self.frame_show_result3, self.frame_show_result4, self.frame_show_result5]
        self.show_images_label_list = [self.label_show_img1, self.label_show_img2, self.label_show_img3, self.label_show_img4, self.label_show_img5]
        self.show_sim_label_list = [self.label_show_sim1, self.label_show_sim2, self.label_show_sim3, self.label_show_sim4, self.label_show_sim5]

        # 获取 widget_show_video
        self.show_video = self.label_7
        # 视频路径和视频捕捉对象
        self.video_path = ""
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_frame)
        self.is_playing = False
        self.current_frame = None

        # 存储检索结果相关变量
        self.current_search_json_result = {}
        self.history_search_json_result = {}

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
        if self.detection_model is not None:
            del self.detection_model
        if self.detection_net is not None:
            del self.detection_net

    # ************************ slot functions ************************ #
    def slot_select_video(self):        
        self.terminal_message("Please select video path")
        # 设置基础路径
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        project_base_path = os.path.abspath(os.path.join(current_file_path, "../data"))
        # print(project_base_path)
        # 打开文件选择对话框
        static_database_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', project_base_path)
        if static_database_file_path:
            self.lineEdit_select_video.setText(static_database_file_path)  # 设置选择的文件路径到 QLineEdit

    def slot_detection_pedestrian(self):      
        # 设置基础路径
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        video_input_path = self.lineEdit_select_video.text()
        video_output_path = os.path.abspath(os.path.join(current_file_path, "../video_out"))
        cropped_img_path = os.path.abspath(os.path.join(current_file_path, "../cropped_imgs"))
        if video_input_path is None:
            # 提示选择视频
            self.terminal_message("Please select video!!", is_error=True)
            return False
        if video_input_path.lower().endswith('.h264') is False:
            # 提示选择.h264文件
            self.terminal_message("Please select '*.h264' file!!", is_error=True)
            return False  
        self.detection_pedestrian(video_input_path, video_output_path, cropped_img_path)

    def slot_select_gt(self):        
        # 设置基础路径
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        project_base_path = os.path.abspath(os.path.join(current_file_path, "../cropped_imgs"))
        # print(project_base_path)
        # 打开文件选择对话框
        gt_imgs_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', project_base_path)
        if gt_imgs_path:
            self.gt_image_path =  gt_imgs_path
            self.show_gt_imgs(gt_imgs_path)
        else:
            self.terminal_message("该图像路径不存在！！")

    def slot_search(self):
        self.terminal_message("=========== Start Search ===========")
        # 获取输入文本描述
        enter_text_description = self.textEdit_enter_text_description.toPlainText()
        if enter_text_description == "":
            self.terminal_message("Please enter text description", is_error=True)
            return
        self.terminal_message("Search style: Dynamic Search")            
        self.terminal_message("Query:")
        self.terminal_message(enter_text_description)            
        if self.get_dynamic_database():
            # 清空结果显示，等待结果
            self.clean_show_result_before_search()
            # 动态检索
            result_sim, result_image_ids, result_image_paths, dataset_base_path = self.dynamic_search(enter_text_description)                
            # 汇总检索结果
            self.catch_json_result(result_sim, result_image_ids, result_image_paths, dataset_base_path)
            # 展示 Top5 结果
            self.show_search_result(self.current_search_json_result)
            # 展示检索结果概要
            self.show_search_result_abstract(self.current_search_json_result)
        else:
            self.terminal_message("ERROR: Dynamic data path dose not contain an image file!", is_error=True)
            return
        
    def slot_clean_terminal_output(self):
        self.textBrowser_terminal_output.clear()

    def slot_open_video_file(self):
        # 打开文件选择对话框
        # video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv)")
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        project_base_path = os.path.abspath(os.path.join(current_file_path, "../video_out"))
        video_path, _ = QFileDialog.getOpenFileName(None, '选择文件', project_base_path)

        if video_path:
            self.video_path = video_path
            self.cap = cv.VideoCapture(self.video_path)
            if self.cap.isOpened():
                # 获取并显示第一帧
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame
                    self.show_first_frame(frame)

    def show_first_frame(self, frame):
        # 将帧转换为QImage格式
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimage = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        # 将第一帧显示在界面上，并且缩小显示，保持比例
        pixmap = QPixmap.fromImage(qimage)
        # 获取 QLabel 当前的尺寸，计算缩放比例
        label_width = self.show_video.width()
        label_height = self.show_video.height()
        # 计算缩放比例
        scale_factor = min(label_width / w, label_height / h)
        # 使用缩放比例调整图片大小
        scaled_pixmap = pixmap.scaled(w * scale_factor, h * scale_factor)
        # 将缩放后的图片显示到界面
        self.show_video.setPixmap(scaled_pixmap)

    def slot_play_video(self):
        if self.cap and self.cap.isOpened():
            self.is_playing = True
            self.timer.start(30)  # 设置定时器每30ms更新一次画面（30帧/秒）

    def slot_pause_video(self):
        self.is_playing = False
        self.timer.stop()  # 停止定时器，暂停视频播放

    def update_video_frame(self):
        if self.is_playing and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # 将当前帧转换为QImage并更新显示
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                qimage = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                # 将当前帧显示到界面上，并且缩小显示，保持比例
                pixmap = QPixmap.fromImage(qimage)
                # 获取 QLabel 当前的尺寸，计算缩放比例
                label_width = self.show_video.width()
                label_height = self.show_video.height()
                # 计算缩放比例
                scale_factor = min(label_width / w, label_height / h)
                # 使用缩放比例调整图片大小
                scaled_pixmap = pixmap.scaled(w * scale_factor, h * scale_factor)
                # 将缩放后的图片显示到界面
                self.show_video.setPixmap(scaled_pixmap)
            else:
                self.timer.stop()  # 视频播放完毕，停止定时器


    # ************************ deploy functions ************************ #
    def detection_pedestrian(self, video_input_path, video_output_path, cropped_img_path):
        self.terminal_message("=========== Start Detection ===========")
        frame_count = 0
        # 打开视频
        video_path = video_input_path
        self.terminal_message("open video: ")
        self.terminal_message(video_path)
        cap = cv.VideoCapture(video_path)
        fps = cap.get(cv.CAP_PROP_FPS)
        Width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        Height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # 获取总帧数
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        total_bar = total_frames + 10  # 进度条的总进度，增加10表示包括后续其他步骤

        # 删除输出目录
        if os.path.exists(video_output_path):
            shutil.rmtree(video_output_path)
        # 删除裁剪图像目录
        if os.path.exists(cropped_img_path):
            shutil.rmtree(cropped_img_path)

        # 创建输出目录
        if not os.path.exists(video_output_path):
            os.mkdir(video_output_path)
        # 创建裁剪图像目录
        if not os.path.exists(cropped_img_path):
            os.mkdir(cropped_img_path)
        output_Video = os.path.basename(video_path)
        output_Video = os.path.join(video_output_path, output_Video)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 视频编码方式
        outVideo = cv.VideoWriter(output_Video, fourcc, fps, (Width, Height))

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

        # 读取视频直到完成
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # 预处理
                data, orig = self.detection_net.preprocess(frame)
                # 发送到模型推理
                result_list = self.detection_model.execute([data,])
                # 处理推理结果
                result_return = self.detection_net.post_process(frame_count, result_list, orig, cropped_img_path)
                # print("result = ", result_return)

                for i in range(len(result_return['detection_classes'])):
                    box = result_return['detection_boxes'][i]
                    class_name = result_return['detection_classes'][i]

                    cv.rectangle(frame, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), colors[i % 6])
                    p3 = (max(int(box[1]), 15), max(int(box[0]), 15))
                    out_label = class_name
                    cv.putText(frame, out_label, p3, cv.FONT_ITALIC, 0.6, colors[i % 6], 1)

                outVideo.write(frame)
                # print("FINISH PROCESSING FRAME: ", frame_count)
                
                # 更新进度条
                self.update_progress_bar_2(frame_count, total_bar)

                frame_count += 1
                # print('\n\n\n')
            else:
                break
        
        self.update_progress_bar_2(total_bar, total_bar)

        cap.release()
        outVideo.release()
        self.terminal_message("Pedestrains Detection Finished")


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
            K = 5
            sorted_indices = np.argsort(similarity, axis=1)[:, ::-1]
            indices = sorted_indices[:, :K]
            top5_values = np.take_along_axis(similarity, indices, axis=1).flatten().tolist()
            top5_indices = np.take_along_axis(index, indices, axis=1).flatten().tolist()
        else:        
            # DEBUG for development on x86
            self.dynamic_image_features = np.random.randn(500, 512)
            N = self.dynamic_image_features.shape[0]
            top5_values = np.random.rand(1, 5).flatten().tolist()
            top5_indices = np.random.randint(0, N, (1, 5)).flatten().tolist()              
        # 5. 返回 Top5 的相似度值和对应的图像路径
        show_images_path =  [os.path.join(dataset_base_path, database_image_files[i]) for i in top5_indices]
        # 6. 设置保存动态图像特征文件名称
        # self.lineEdit_dynamic_to_static_name.setText(f"{self.dynamic_dataset_folder_name}_test_data.npy")
        self.update_progress_bar(total_bar, total_bar)
        return top5_values, top5_indices, show_images_path, dataset_base_path



    # ************************ utils functions ************************ #
    def terminal_message(self, text, is_error=False):
        if is_error:
            self.textBrowser_terminal_output.append(f"<span style='color:red;'>{text}</span>")
        else:
            self.textBrowser_terminal_output.append(f"<span style='color:black;'>{text}</span>")
        self.textBrowser_terminal_output.moveCursor(self.textBrowser_terminal_output.textCursor().End)

    # 显示 gt
    def show_gt_imgs(self, gt_imgs_path):
        if os.path.exists(gt_imgs_path) is False:
            # 提示选择数据集
            self.terminal_message("Please select true gt data path", is_error=True)
            return False
        pixmap = QPixmap(gt_imgs_path)
        resized_pixmap = pixmap.scaled(90, 120)                     
        self.label_show_gt.setPixmap(resized_pixmap) 
        self.label_show_gt.setScaledContents(True)                                   
        self.label_show_gt.setAlignment(QtCore.Qt.AlignCenter) 

    def get_dynamic_database(self):
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        dynamic_database_path = os.path.abspath(os.path.join(current_file_path, "../cropped_imgs"))
        # dynamic_database_path = self.lineEdit_select_path.text()
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
        result_sim = show_search_json_result["similarity"]
        result_image_paths = show_search_json_result["image_paths"]
        dataset_base_path = show_search_json_result["dataset_base_path"]
        result_pids = show_search_json_result["result_imgids_or_pids"]
        pids = show_search_json_result["set_pid"]           
        # 展示 Top5 图像及相似度
        for i in range(5):
            image_path = os.path.join(dataset_base_path, result_image_paths[i])
            sim = result_sim[i] 
            pixmap = QPixmap(image_path)            
            resized_pixmap = pixmap.scaled(140, 220) 
            self.show_images_label_list[i].setPixmap(resized_pixmap)
            self.show_sim_label_list[i].setText(f"匹配度: {sim:.3f}")            
            self.show_images_label_list[i].setAlignment(QtCore.Qt.AlignCenter)
            self.show_sim_label_list[i].setAlignment(QtCore.Qt.AlignCenter)            
        # 设置 frame 边框显示   
        for i in range(5):
            self.show_frame_border(i, show_border=False) 

    def update_progress_bar(self, i, N):
        value = int(i / N * 100)
        # 更新进度条
        self.progressBar.setValue(value)

    def update_progress_bar_2(self, i, N):
        value = int(i / N * 100)
        # 更新进度条
        self.progressBar_2.setValue(value)

    def catch_json_result(self, result_sim, result_imgids_or_pids, result_image_paths, dataset_base_path):
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
        if os.path.exists(self.gt_image_path) is False:
            # 提示选择数据集
            self.terminal_message("Please select true gt data path", is_error=True)
            self.gt_image_path = os.path.join(current_file_path, "ui_data/ui_no_gt.jpg")
            # return False
        pixmap = QPixmap(self.gt_image_path)
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
        # 清空 Top5 结果
        for i in range(5):
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

    # def play_video(self, video_path):
    #     # 使用 FFmpeg 解码视频（OpenCV 使用 FFmpeg）
    #     cap = cv.VideoCapture(video_path)

    #     if not cap.isOpened():
    #         print("Error: Could not open video.")
    #         return

    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         # 将 OpenCV 帧转换为 QImage 格式
    #         frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #         h, w, ch = frame_rgb.shape
    #         qimage = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)

    #         # 将 QImage 显示到 QLabel 上
    #         self.show_video.setPixmap(QPixmap.fromImage(qimage))
    #         QApplication.processEvents()  # 更新界面，避免卡死

    #     cap.release()

    def play_video(self, video_path):
        # 使用 FFmpeg 解码视频（OpenCV 使用 FFmpeg）
        self.cap = cv.VideoCapture(video_path)

        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        # 播放视频
        self.is_playing = True
        while self.cap.isOpened() and self.is_playing:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 将 OpenCV 帧转换为 QImage 格式
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimage = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)

            # 将 QImage 显示到 QLabel 上
            self.show_video.setPixmap(QPixmap.fromImage(qimage))
            QApplication.processEvents()  # 更新界面，避免卡死

        self.cap.release()