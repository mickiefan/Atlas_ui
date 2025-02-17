import sys
import os
import numpy as np
import cv2 as cv
from PIL import Image
from acllite.acllite_model import AclLiteModel
from acllite.acllite_resource import AclLiteResource

labels = ["person",
        "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
        "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]
OUTPUT_DIR = '../video_out/'
CROPPRD_OUTPUT_DIR = '../cropped_imgs/'
MODEL_PATH = "../model/yolov4_bs1.om"
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

class detection_net:
    def __init__(self, model_path):    
        #ACL resource initialization
        acl_resource = AclLiteResource()
        acl_resource.init()
        #load model
        self.model = AclLiteModel(model_path)

    def preprocess(self, frame):
        image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        img_h = image.size[1]
        img_w = image.size[0]
        net_h = MODEL_HEIGHT
        net_w = MODEL_WIDTH

        scale = min(float(net_w) / float(img_w), float(net_h) / float(img_h))
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        shift_x = (net_w - new_w) // 2
        shift_y = (net_h - new_h) // 2
        shift_x_ratio = (net_w - new_w) / 2.0 / net_w
        shift_y_ratio = (net_h - new_h) / 2.0 / net_h

        image_ = image.resize((new_w, new_h))
        new_image = np.zeros((net_h, net_w, 3), np.uint8)
        new_image[shift_y: new_h + shift_y, shift_x: new_w + shift_x, :] = np.array(image_)
        new_image = new_image.astype(np.float32)
        new_image = new_image / 255
        print('new_image.shape', new_image.shape)
        new_image = new_image.transpose(2, 0, 1).copy()
        return new_image, image

    def overlap(self, x1, x2, x3, x4):
        left = max(x1, x3)
        right = min(x2, x4)
        return right - left

    def cal_iou(self, box, truth):
        w = self.overlap(box[0], box[2], truth[0], truth[2])
        h = self.overlap(box[1], box[3], truth[1], truth[3])
        if w <= 0 or h <= 0:
            return 0
        inter_area = w * h
        union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
        return inter_area * 1.0 / union_area

    def apply_nms(self, all_boxes, thres):
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
                    iou = self.cal_iou(box, truth)
                    if iou >= thres:
                        p[j] = 1
            for i in range(len(sorted_boxes)):
                if i not in p:
                    res.append(sorted_boxes[i])
        return res

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def decode_bbox(self, conv_output, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio):
        print('conv_output.shape', conv_output.shape)
        _, _, h, w = conv_output.shape 
        conv_output = conv_output.transpose(0, 2, 3, 1)
        pred = conv_output.reshape((h * w, 3, 5 + class_num))
        pred[..., 4:] = self._sigmoid(pred[..., 4:])
        pred[..., 0] = (self._sigmoid(pred[..., 0]) + np.tile(range(w), (3, h)).transpose((1, 0))) / w
        pred[..., 1] = (self._sigmoid(pred[..., 1]) + np.tile(np.repeat(range(h), w), (3, 1)).transpose((1, 0))) / h
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
        pred = pred[pred[:, 4] >= 0.6]
        print('pred[:, 5]', pred[:, 5])
        print('pred[:, 5] shape', pred[:, 5].shape)

        all_boxes = [[] for ix in range(class_num)]
        for ix in range(pred.shape[0]):
            box = [int(pred[ix, iy]) for iy in range(4)]
            box.append(int(pred[ix, 5]))
            box.append(pred[ix, 4])
            all_boxes[box[4] - 1].append(box)
        return all_boxes

    def convert_labels(self, label_list):
        if isinstance(label_list, np.ndarray):
            label_list = label_list.tolist()
            label_names = [labels[int(index)] for index in label_list]
        return label_names

    def post_process(self, frame_count, infer_output, origin_img, cropped_img_path):
        print("post process")
        result_return = dict()
        img_h = origin_img.size[1]
        img_w = origin_img.size[0]
        scale = min(float(MODEL_WIDTH) / float(img_w), float(MODEL_HEIGHT) / float(img_h))
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        shift_x_ratio = (MODEL_WIDTH - new_w) / 2.0 / MODEL_WIDTH
        shift_y_ratio = (MODEL_HEIGHT - new_h) / 2.0 / MODEL_HEIGHT
        class_number = len(labels)
        num_channel = 3 * (class_number + 5)
        x_scale = MODEL_WIDTH / float(new_w)
        y_scale = MODEL_HEIGHT / float(new_h)
        all_boxes = [[] for ix in range(class_number)]
        print(infer_output[0].shape)
        print(infer_output[1].shape)
        print(infer_output[2].shape)

        for ix in range(3):    
            pred = infer_output[ix]
            print('pred.shape', pred.shape)
            anchors = anchor_list[ix]
            boxes = self.decode_bbox(pred, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio)
            all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_number)]

        res = self.apply_nms(all_boxes, iou_threshold)

        # 只保留类别为"person"的框，并裁剪图像
        person_boxes = [box for box in res if box[4] == 0]  # 1是“person”类的索引
        cropped_images = []

        for idx, box in enumerate(person_boxes):
            x1, y1, x2, y2 = box[:4]
            cropped_img = origin_img.crop((x1, y1, x2, y2))
            cropped_images.append(cropped_img)

            # 保存裁剪后的图像
            output_path = os.path.join(cropped_img_path, f"cropped_person_{frame_count}_{idx + 1}.jpg")
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
            picked_classes = self.convert_labels(new_res[:, 4])
            picked_score = new_res[:, 5]
            result_return['detection_classes'] = picked_classes
            result_return['detection_boxes'] = picked_boxes.tolist()
            result_return['detection_scores'] = picked_score.tolist()
            return result_return

def main():
    if (len(sys.argv) != 2):
        print("Please input video path")
        exit(1)
    frame_count = 0
    #open video
    video_path = sys.argv[1]
    print("open video ", video_path)
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    Width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    Height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    #create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    output_Video = os.path.basename(video_path)
    output_Video = os.path.join(OUTPUT_DIR, output_Video)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # DIVX, XVID, MJPG, X264, WMV1, WMV2
    outVideo = cv.VideoWriter(output_Video, fourcc, fps, (Width, Height))

    # Initialize model
    dection_net = detection_net(MODEL_PATH)


    # Read until video is completed
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            #preprocess
            data, orig = dection_net.preprocess(frame)
            #Send into model inference
            result_list = dection_net.model.execute([data,])
            #Process inference results
            result_return = dection_net.post_process(frame_count, result_list, orig, CROPPRD_OUTPUT_DIR)
            print("result = ", result_return)

            for i in range(len(result_return['detection_classes'])):
                box = result_return['detection_boxes'][i]
                class_name = result_return['detection_classes'][i]
                confidence = result_return['detection_scores'][i]
                # distance[i] = calculate_position(bbox=box, transform_matrix=perspective_transform,
                #             warped_size=WARPED_SIZE, pix_per_meter=pixels_per_meter)
                # label_dis = '{} {:.2f}m'.format('dis:', distance[i][0])
                # cv.putText(frame, label_dis, (int(box[1]) + 10, int(box[2]) + 15), 
                #             cv.FONT_ITALIC, 0.6, colors[i % 6], 1)

                cv.rectangle(frame, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), colors[i % 6])
                p3 = (max(int(box[1]), 15), max(int(box[0]), 15))
                out_label = class_name
                cv.putText(frame, out_label, p3, cv.FONT_ITALIC, 0.6, colors[i % 6], 1)

            outVideo.write(frame)
            print("FINISH PROCESSING FRAME: ", frame_count)
            frame_count += 1
            print('\n\n\n')
        # Break the loop
        else:
            break
    cap.release()
    outVideo.release()
    print("Execute end")


if __name__ == '__main__':
    main()
