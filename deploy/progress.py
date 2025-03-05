import numpy as np
from PIL import Image
import cv2 as cv
import os

labels =["person",  "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench","bird", 
        "cat", "dog", "horse", "sheep", "cow", 
        "elephant",  "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag",  "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball",  "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

MODEL_WIDTH = 608
MODEL_HEIGHT = 608
class_num = 80
stride_list = [32, 16, 8]
anchors_3 = np.array([[12, 16], [19, 36], [40, 28]]) / stride_list[2]
anchors_2 = np.array([[36, 75], [76, 55], [72, 146]]) / stride_list[1]
anchors_1 = np.array([[142, 110], [192, 243], [459, 401]]) / stride_list[0]
anchor_list = [anchors_1, anchors_2, anchors_3]
iou_threshold = 0.3
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

# def preprocess(frame):
#     # 将 AclLiteImage 转换为 NumPy 数组
#     frame_np = frame.byte_data_to_np_array().astype(np.uint8)
    
#     # 获取图像的高度和宽度
#     height, width = map(int, (frame.height, frame.width))
    
#     # 计算各平面的大小
#     y_size = height * width
#     uv_size = (height // 2) * (width // 2)

#     # 分离 Y、U、V 平面
#     y = frame_np[:y_size].reshape((height, width))
#     u = frame_np[y_size:y_size + uv_size].reshape((height // 2, width // 2))
#     v = frame_np[y_size + uv_size:].reshape((height // 2, width // 2))

#     # 上采样 U 和 V 平面
#     u_upscaled = cv.resize(u, (width, height), interpolation=cv.INTER_LINEAR)
#     v_upscaled = cv.resize(v, (width, height), interpolation=cv.INTER_LINEAR)

#     # YUV420P -> RGB conversion manually
#     # YUV to RGB conversion using the corrected formula
#     rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

#     rgb_image[:, :, 0] = np.clip(y + 1.402 * (v_upscaled - 128), 0, 255)  # Red
#     rgb_image[:, :, 1] = np.clip(y - 0.344136 * (u_upscaled - 128) - 0.714136 * (v_upscaled - 128), 0, 255)  # Green
#     rgb_image[:, :, 2] = np.clip(y + 1.772 * (u_upscaled - 128), 0, 255)  # Blue

#     # Convert to PIL image for visualization
#     image = Image.fromarray(rgb_image)

#     # Return the RGB image (no further processing)
#     return image

def overlap(x1, x2, x3, x4):
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(all_boxes, thres):
    res = []
    for cls in range(class_num):
        cls_bboxes = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]
        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue
            truth = sorted_boxes[i]
            for j in range(i + 1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1
        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res

def _sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def decode_bbox(conv_output, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio):
    print('conv_output.shape', conv_output.shape)
    _, _, h, w = conv_output.shape 
    conv_output = conv_output.transpose(0, 2, 3, 1)
    pred = conv_output.reshape((h * w, 3, 5 + class_num))
    pred[..., 4:] = _sigmoid(pred[..., 4:])
    pred[..., 0] = (_sigmoid(pred[..., 0]) + np.tile(range(w), (3, h)).transpose((1, 0))) / w
    pred[..., 1] = (_sigmoid(pred[..., 1]) + np.tile(np.repeat(range(h), w), (3, 1)).transpose((1, 0))) / h
    pred[..., 2] = np.exp(pred[..., 2]) * anchors[:, 0:1].transpose((1, 0)) / w
    pred[..., 3] = np.exp(pred[..., 3]) * anchors[:, 1:2].transpose((1, 0)) / h

    bbox = np.zeros((h * w, 3, 4))
    bbox[..., 0] = np.maximum((pred[..., 0] - pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, 0)  # x_min
    bbox[..., 1] = np.maximum((pred[..., 1] - pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, 0)  # y_min
    bbox[..., 2] = np.minimum((pred[..., 0] + pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, img_w)  # x_max
    bbox[..., 3] = np.minimum((pred[..., 1] + pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, img_h)  # y_max
    pred[..., :4] = bbox
    pred = pred.reshape((-1, 5 + class_num))
    pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)
    pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)    
    pred = pred[pred[:, 4] >= 0.8] # 增大筛选阈值

    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
        box = [int(pred[ix, iy]) for iy in range(4)]
        box.append(int(pred[ix, 5]))
        box.append(pred[ix, 4])
        all_boxes[box[4] - 1].append(box)
    return all_boxes

def convert_labels(label_list):
    if isinstance(label_list, np.ndarray):
        label_list = label_list.tolist()
        label_names = [labels[int(index)] for index in label_list]
    return label_names

def post_process(infer_output, rgb_image, OUTPUT_DIR, frame_count):
    result_return = dict()
    img_h = MODEL_HEIGHT
    img_w = MODEL_WIDTH
    scale = min(float(MODEL_WIDTH) / float(img_w), float(MODEL_HEIGHT) / float(img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    shift_x_ratio = (MODEL_WIDTH - new_w) / 2.0 / MODEL_WIDTH
    shift_y_ratio = (MODEL_HEIGHT - new_h) / 2.0 / MODEL_HEIGHT
    class_number = len(labels)
    x_scale = MODEL_WIDTH / float(new_w)
    y_scale = MODEL_HEIGHT / float(new_h)
    all_boxes = [[] for ix in range(class_number)]
    for ix in range(3):    
        pred = infer_output[ix]
        print('pred.shape', pred.shape)
        anchors = anchor_list[ix]
        boxes = decode_bbox(pred, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio)
        all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_number)]

    res = apply_nms(all_boxes, iou_threshold)

    # 只保留类别为"person"的框，并裁剪图像
    person_boxes = [box for box in res if box[4] == 0]  # 1是“person”类的索引
    cropped_images = []

    # 每一秒(30帧)保存一次行人图像
    if (frame_count % 30 == 0):
        for idx, box in enumerate(person_boxes):
            x1, y1, x2, y2 = box[:4]
            cropped_img = rgb_image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped_img)

            # 保存裁剪后的图像
            output_path = os.path.join(OUTPUT_DIR, f"cropped_person_{frame_count}_{idx + 1}.jpg")
            cropped_img.save(output_path)

    if not person_boxes:
        result_return['detection_classes'] = []
        result_return['detection_boxes'] = []
        result_return['detection_scores'] = []
        return result_return
    else:
        new_res = np.array(person_boxes)
        picked_boxes = new_res[:, 0:4]
        picked_boxes = picked_boxes[:, [1, 0, 3, 2]]
        picked_classes = convert_labels(new_res[:, 4])
        picked_score = new_res[:, 5]
        result_return['detection_classes'] = picked_classes
        result_return['detection_boxes'] = picked_boxes.tolist()
        result_return['detection_scores'] = picked_score.tolist()
        return result_return

def yuv420sp_to_rgb(yuv420sp, width, height):
    # 检查数据长度是否符合 YUV420SP (NV12) 格式
    # expected_size = width * height * 3 // 2
    # if len(yuv420sp) != expected_size:
    #     raise ValueError(f"Invalid YUV data size: expected {expected_size}, got {len(yuv420sp)}")

    # 转换 numpy 数组
    yuv420sp = np.frombuffer(yuv420sp, dtype=np.uint8)

    # 分离 Y 和 UV 分量
    y_size = width * height
    y = yuv420sp[:y_size].reshape((height, width))

    # 提取 UV 数据并拆分成 U 和 V
    uv = yuv420sp[y_size:].reshape((height // 2, width))  # 交错 UV 格式
    u = uv[:, 0::2]  # 取偶数列 (U)
    v = uv[:, 1::2]  # 取奇数列 (V)

    # 使用更平滑的插值方式进行 U/V 通道上采样
    u = cv.resize(u, (width, height), interpolation=cv.INTER_LINEAR)
    v = cv.resize(v, (width, height), interpolation=cv.INTER_LINEAR)

    # 合并 YUV 分量
    yuv = np.stack((y, u, v), axis=-1)

    # YUV 转 RGB（可尝试不同颜色空间）
    rgb = cv.cvtColor(yuv, cv.COLOR_YUV2RGB)

    return rgb

# 弃用: 速度太慢
# def yuv420sp_to_jpeg(filename, pdata, image_width, image_height, quality):
#     # 将 pdata 转换为 numpy 数组
#     y_size = image_width * image_height
#     y_data = np.frombuffer(pdata[:y_size], dtype=np.uint8).reshape((image_height, image_width))
#     uv_data = np.frombuffer(pdata[y_size:], dtype=np.uint8).reshape((image_height // 2, image_width // 2, 2))

#     # 创建 RGB 图像
#     rgb_data = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    
#     # 填充 RGB 数据
#     for y in range(image_height):
#         for x in range(image_width):
#             Y = y_data[y, x]
#             u = uv_data[y // 2, x // 2, 0]
#             v = uv_data[y // 2, x // 2, 1]
            
#             # YUV to RGB 转换公式
#             R = Y + 1.402 * (v - 128)
#             G = Y - 0.344136 * (u - 128) - 0.714136 * (v - 128)
#             B = Y + 1.772 * (u - 128)
            
#             # 限制 RGB 值在 0-255 范围内
#             rgb_data[y, x, 0] = np.clip(R, 0, 255)
#             rgb_data[y, x, 1] = np.clip(G, 0, 255)
#             rgb_data[y, x, 2] = np.clip(B, 0, 255)

#     # 使用 PIL 保存为 JPEG 文件
#     img = Image.fromarray(rgb_data)
#     img.save(filename, 'JPEG', quality=quality)

def deal_result(result_return, rgb_image):
# -------------------------------------------------------
    # # 假设 yuv_data 是包含 YUV420SP 数据的字节流
    # # yuv_data = None  # 替换为实际的 YUV 数据
    # image_width = 608  # 图像宽度
    # image_height = 608  # 图像高度
    # output_filename = "output.jpg"
    # quality = 10  # JPEG 图像质量 (0-100)

    # # 调用函数将 YUV420SP 转换为 JPEG
    # yuv420sp_to_jpeg(output_filename, yuv_data, image_width, image_height, quality)
# -------------------------------------------------------
    for i in range(len(result_return['detection_classes'])):
        box = result_return['detection_boxes'][i]
        class_name = result_return['detection_classes'][i]
        cv.rectangle(rgb_image, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), colors[i % 6])
        p3 = (max(int(box[1]), 15), max(int(box[0]), 15))
        out_label = class_name            
        cv.putText(rgb_image, out_label, p3, cv.FONT_ITALIC, 0.6, colors[i % 6], 1)
    return rgb_image
