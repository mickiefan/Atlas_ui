import os
import sys
# 通过当前文件目录的相对路径设置工程的根目录
current_file_path = os.path.abspath(os.path.dirname(__file__))
project_base_path = os.path.abspath(os.path.join(current_file_path, "../"))
sys.path.append(project_base_path)

from config import DEVICE_IS_ATLAS
if DEVICE_IS_ATLAS:
    import acl

import json
import numpy as np
from PIL import Image




ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def check_ret(message, ret):
    """检查返回数据"""
    if ret != ACL_SUCCESS:
        raise Exception("{} failed ret={}"
                        .format(message, ret))

class net:
    def __init__(self, model_path):
        self.device_id = 0 
        # step 1: 初始化
        ret = acl.init()
        # print("init resource stage:")
        # 制定运算的 device
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        # setp 2: 加载模型
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        check_ret("acl.mdl.load_from_file", ret)
        # 创建空白模型描述信息，获取模型的描述信息的指针地址
        self.model_desc = acl.mdl.create_desc()
        # 通过模型的 ID，将模型的描述信息填充到 mdoel_desc 中
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)

        # step 3: 创建输入输出数据集
        # 创建输入数据集
        self.input_dataset, self.input_data = self.prepare_dataset('input')
        # 创建输出数据集
        self.output_dataset, self.output_data = self.prepare_dataset('output')

    def prepare_dataset(self, io_type):
        # 准备数据集
        if io_type == "input":
            # 获得模型输入的个数
            io_num = acl.mdl.get_num_inputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_input_size_by_index
        else:
            # 获得模型输出的个数
            io_num = acl.mdl.get_num_outputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_output_size_by_index
        # 创建aclmdlDataset类型的数据，描述模型推理的输入。
        dataset = acl.mdl.create_dataset()
        datas = []
        for i in range(io_num):
            # 获取所需的buffer内存大小
            buffer_size = acl_mdl_get_size_by_index(self.model_desc, i)
            # 申请buffer内存
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            # 从内存创建buffer数据
            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            # 将buffer数据添加到数据集
            _, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
            datas.append({"buffer": buffer, "data": data_buffer, "size": buffer_size})
        return dataset, datas

    def image_forward(self, inputs):
        # 遍历所有输入，拷贝到对应的 buffer 内存中
        input_num = len(inputs)
        # print("input num:", input_num)
        for i in range(input_num):
            bytes_data = inputs[i].tobytes()
            bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
            # 将图像数据从 Host 传输到 Device
            ret = acl.rt.memcpy(self.input_data[i]["buffer"], # 目标地址 device
                                self.input_data[i]["size"], # 目标地址大小
                                bytes_ptr, # 源地址 host
                                len(bytes_data), # 源地址大小
                                ACL_MEMCPY_HOST_TO_DEVICE)
        # 执行推理
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        # 处理模型推理的输出数据，输出 top5 置信度的类别编号
        inference_result = []
        for i, item in enumerate(self.output_data):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            # 将 buffer 数据从 Device 传输到 Host
            ret = acl.rt.memcpy(buffer_host, 
                                self.output_data[i]["size"],
                                self.output_data[i]["buffer"], 
                                self.output_data[i]["size"],
                                ACL_MEMCPY_DEVICE_TO_HOST)
            # 从内存地址获取 bytes 对象
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            # 按照 float32 格式将数据转换为 numpy 数组
            data = np.frombuffer(bytes_out, dtype=np.float32)
            inference_result.append(data.reshape(-1, 512))
        return inference_result[0]

    def text_forward(self, inputs):
        # 遍历所有输入，拷贝到对应的 buffer 内存中
        input_num = len(inputs)
        # print("input num:", input_num)
        for i in range(input_num):
            bytes_data = inputs[i].tobytes()
            bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
            # 将图像数据从 Host 传输到 Device
            ret = acl.rt.memcpy(self.input_data[i]["buffer"], # 目标地址 device
                                self.input_data[i]["size"], # 目标地址大小
                                bytes_ptr, # 源地址 host
                                len(bytes_data), # 源地址大小
                                ACL_MEMCPY_HOST_TO_DEVICE)
        # 执行推理
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        # 处理模型推理的输出数据，输出 top5 置信度的类别编号
        inference_result = []
        for i, item in enumerate(self.output_data):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            # 将 buffer 数据从 Device 传输到 Host
            ret = acl.rt.memcpy(buffer_host, 
                                self.output_data[i]["size"],
                                self.output_data[i]["buffer"], 
                                self.output_data[i]["size"],
                                ACL_MEMCPY_DEVICE_TO_HOST)
            # 从内存地址获取 bytes 对象
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            # 按照 float32 格式将数据转换为 numpy 数组
            data = np.frombuffer(bytes_out, dtype=np.float32)
            inference_result.append(data.reshape(-1, 512))
        return inference_result[0]

    def similarity_forward(self, inputs):
        # 遍历所有输入，拷贝到对应的 buffer 内存中
        input_num = len(inputs)
        # print("input num:", input_num)
        for i in range(input_num):
            bytes_data = inputs[i].tobytes()
            bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
            # 将图像数据从 Host 传输到 Device
            ret = acl.rt.memcpy(self.input_data[i]["buffer"], # 目标地址 device
                                self.input_data[i]["size"], # 目标地址大小
                                bytes_ptr, # 源地址 host
                                len(bytes_data), # 源地址大小
                                ACL_MEMCPY_HOST_TO_DEVICE)
        # 执行推理
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        # 处理模型推理的输出数据，输出 top5 置信度的类别编号
        inference_result = []
        for i, item in enumerate(self.output_data):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            # 将 buffer 数据从 Device 传输到 Host
            ret = acl.rt.memcpy(buffer_host, 
                                self.output_data[i]["size"],
                                self.output_data[i]["buffer"], 
                                self.output_data[i]["size"],
                                ACL_MEMCPY_DEVICE_TO_HOST)
            # 从内存地址获取 bytes 对象
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            if i == 0:
                # 按照 float32 格式将数据转换为 numpy 数组
                data = np.frombuffer(bytes_out, dtype=np.float32)
            else:
                # 索引数据
                data = np.frombuffer(bytes_out, dtype=np.int64)
            inference_result.append(data.reshape(1, -1))
        return inference_result

    def __del__(self):
        # 销毁输入输出数据集
        for dataset in [self.input_data, self.output_data]:
            while dataset:
                item = dataset.pop()
                ret = acl.destroy_data_buffer(item["data"]) # 销毁 buffer 数据
                ret = acl.rt.free(item["buffer"]) # 释放 buffer 内存
        ret = acl.mdl.destroy_dataset(self.input_dataset) # 销毁输入数据集
        ret = acl.mdl.destroy_dataset(self.output_dataset) # 销毁输出数据集
        # 销毁模型描述信息
        ret = acl.mdl.destroy_desc(self.model_desc)
        # 卸载模型
        ret = acl.mdl.unload(self.model_id)
        # 释放 device
        ret = acl.rt.reset_device(self.device_id)
        # acl 去初始化
        ret = acl.finalize()


def transfer_pic(input_path):
    # 图像预处理
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    mean = np.expand_dims(mean, axis=(1, 2)).astype(np.float32)
    std = np.expand_dims(std, axis=(1, 2)).astype(np.float32)

    with Image.open(input_path) as image_file:
        img = image_file.convert('RGB')
        # img = img.resize((128, 384)) # 有精度损失
        img = img.resize((128, 384), Image.BILINEAR)                
        img = np.array(img).astype(np.float32) / 255.0
    # HWC -> CHW 
    img = img.transpose((2, 0, 1))      
    img = (img - mean) / std    

    return np.array([img])

def tokenize(caption: str, tokenizer, text_length=77, truncate=True):
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = np.zeros(text_length, dtype=np.int64)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = tokens
    return result



if __name__ == "__main__":    
    
    from simple_tokenizer import SimpleTokenizer

    data_root = "/home/lugengyou/workspace/Codes/lugy_research/distil_tbps/data"
    model_root = "/home/lugengyou/workspace/Codes/lugy_research/distil_tbps/deploy/CUHK-PEDES/model"

    # 1. 加载图像数据库特征，及图像路径
    test_dataset = json.load(open(os.path.join(data_root, "CUHK-PEDES_test_data.json")))
    test_image_norm_features = np.load(os.path.join(data_root, "CUHK-PEDES_test_image_norm_database.npy"))
    images_paths = test_dataset["img_paths"]
    images_pids = test_dataset["image_pids"]
    N = len(images_paths)

    # 2. 初始化文本编码器模型
    bpe_path = os.path.join(data_root, "bpe_simple_vocab_16e6.txt.gz")
    tokenizer = SimpleTokenizer(bpe_path)
    text_encoder = net(os.path.join(model_root, "xsmall_text_encode_910B.om"))

    # 3. 初始化相似度计算模型
    consine_sim_model = net(os.path.join(model_root, "similarity_910B.om"))

    # 4. 接受文本输入，进行文本编码 
    text = "The man has short, dark hair and wears khaki pants with an oversized grey hoodie. His black backpack hangs from one shoulder."
    text = tokenize(text, tokenizer=tokenizer, text_length=77, truncate=True)
    text = text.reshape((1, 77))
    result = text_encoder.text_forward(text) # npu 计算     
    text_feature = result[text.argmax(axis=-1), :] # 获取最大值的索引对应的特征，即为文本的 cls 特征
    
    # 5. 计算图像数据库特征与文本特征的相似度
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
        result = consine_sim_model.similarity_forward(inputs) # npu 计算  
        similarity.append(result[0])
        index.append(result[1])        
    # 处理不整除的情况
    if N % 1024 != 0:
        start_index = loops * 1024
        images = np.zeros((1024, 512), dtype=np.float32)
        images[0 : N - start_index] = test_image_norm_features[start_index:]
        start_index = np.array([start_index], dtype=np.int64)
        inputs = [images, text_feature, start_index]
        result = consine_sim_model.similarity_forward(inputs)
        similarity.append(result[0])
        index.append(result[1])

    # 5. 合并结果,并进行最终 TopK 操作    
    similarity = np.concatenate(similarity, axis=1)
    index = np.concatenate(index, axis=1)    
    # 获取前 K 个最大值的索引
    K = 10
    sorted_indices = np.argsort(similarity, axis=1)[:, ::-1]
    indices = sorted_indices[:, :K]
    top10_values = np.take_along_axis(similarity, indices, axis=1)
    top10_indices = np.take_along_axis(index, indices, axis=1)
    print("numpy similarity output:", top10_values, top10_indices)    

    del consine_sim_model
    del text_encoder
