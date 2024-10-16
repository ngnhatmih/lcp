# license_plate_detection.py

import cv2
import numpy as np
import math
import pytesseract
from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt

CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="8PKnBAYVuzxib2KznYlV")

def display(im_data):
    """Hiển thị ảnh."""
    dpi = 80
    height, width  = im_data.shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.show()

def get_plate_predictions(file, model_id="plate-detect-bkwoo/1"):
    """Lấy dự đoán từ model."""
    result = CLIENT.infer(file, model_id=model_id)
    return result.get('predictions', [])

def crop_plate(img, plate):
    """Cắt vùng chứa biển số từ ảnh."""
    x_0, y_0, w, h = plate['x'], plate['y'], plate['width'], plate['height']
    x_1, y_1 = x_0 - w / 2, y_0 - h / 2
    x_2, y_2 = x_1 + w, y_1 + h
    x1, y1, x2, y2 = (int(x_1), int(y_1), int(x_2), int(y_2))
    crop = img[y1-12:y2+1, x1:x2]
    resize = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return resize

def compute_skew(image):
    """Tính toán góc nghiêng của ảnh."""
    h, w, _ = image.shape
    img = cv2.medianBlur(image, 3)
    edges = cv2.Canny(img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 4.0, maxLineGap=h/4.0)
    
    if lines is None:
        return 0.0
    
    angle = 0.0
    cnt = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            ang = np.arctan2(y2 - y1, x2 - x1)
            if math.fabs(ang) <= 30:
                angle += ang
                cnt += 1
    return (angle / cnt) * 180 / math.pi if cnt != 0 else 0.0

def deskew(image):
    """Xoay ảnh về trạng thái cân bằng."""
    angle = compute_skew(image)
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def preprocess_image(image):
    """Tiền xử lý ảnh."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    gamma = adjust_gamma(blur, gamma=0.5)
    _, thresh = cv2.threshold(gamma, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 4))
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)
    return dilation

def adjust_gamma(image, gamma=1.0):
    """Điều chỉnh độ sáng của ảnh để chống lóa."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def extract_characters(dilation, rotated):
    """Trích xuất các ký tự từ ảnh biển số đã xử lý."""
    cnts, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plate_cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[1]
    cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

    height = cv2.boundingRect(plate_cnt)[-1]
    width = cv2.boundingRect(plate_cnt)[-2]
    top_half, bottom_half, rois = [], [], []
    
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if height / h > 4 or height / h < 1.5:
            continue
        roi = dilation[y-5:y + h + 5, x-5:x + w + 5]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)
        if width / height >= 3.5:
            rois.append(roi)
        else:
            if y < height / 2 - 20:
                top_half.append(roi)
            else:
                bottom_half.append(roi)
    
    return rois if (top_half and bottom_half) else top_half + bottom_half

def recognize_plate_characters(rois):
    """Sử dụng OCR để nhận diện ký tự trên biển số."""
    plate_num = ''
    max_height = max(roi.shape[0] for roi in rois)
    _rois = [cv2.resize(roi, (roi.shape[1], max_height)) for roi in rois]
    for roi in _rois:
        char = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3').strip()
        plate_num += char
    return plate_num

def detect_license_plate(file):
    """Chuỗi xử lý chính để phát hiện và nhận dạng biển số xe."""
    plates = get_plate_predictions(file)
    if not plates:
        return "Không tìm thấy biển số"

    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_plate = crop_plate(img, plates[0])
    rotated = deskew(resized_plate)
    processed = preprocess_image(rotated)
    rois = extract_characters(processed, rotated)
    plate_num = recognize_plate_characters(rois)
    return plate_num
