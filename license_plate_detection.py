import cv2
import numpy as np
import math
import pytesseract
from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt
import os

CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com",
                             api_key="8PKnBAYVuzxib2KznYlV")


def save_image(image, filename):
    """Lưu ảnh với tên tệp chỉ định."""
    if not os.path.exists("processed_images"):
        os.makedirs("processed_images")
    cv2.imwrite(f"processed_images/{filename}", image)


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
    crop = img[y1:y2, x1 - 5:x2 + 5]
    resize = cv2.resize(crop, None, fx=5, fy=3, interpolation=cv2.INTER_CUBIC)
    save_image(resize, "resized_plate.jpg")
    return resize


def compute_skew(image):
    h, w, _ = image.shape

    img = cv2.medianBlur(image, 3)

    edges = cv2.Canny(img,
                      threshold1=30,
                      threshold2=100,
                      apertureSize=3,
                      L2gradient=True)
    lines = cv2.HoughLinesP(edges,
                            1,
                            math.pi / 180,
                            30,
                            minLineLength=w / 4.0,
                            maxLineGap=h / 4.0)
    angle = 0.0
    nlines = lines.size

    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30:
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0

    if math.fabs((angle / cnt) * 180 / math.pi) == 90:
        return 0.0

    return (angle / cnt) * 180 / math.pi


def deskew(image):
    """Xoay ảnh về trạng thái cân bằng."""
    angle = compute_skew(image)
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image,
                              rot_mat,
                              image.shape[1::-1],
                              flags=cv2.INTER_LINEAR)
    save_image(deskewed, "deskewed_plate.jpg")
    return deskewed


def adjust_gamma(image, gamma=1.0):
    """Điều chỉnh độ sáng của ảnh để chống lóa."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0)**inv_gamma * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def stretch_image(image, scale_x=1.5, scale_y=1.5):
    """Kéo giãn ảnh theo cả chiều ngang và chiều dọc."""
    original_height, original_width = image.shape[:2]

    new_width = int(original_width * scale_x)
    new_height = int(original_height * scale_y)
    new_size = (new_width, new_height)

    stretched_image = cv2.resize(image,
                                 new_size,
                                 interpolation=cv2.INTER_LINEAR)
    return stretched_image


def preprocess_image(image):
    """Tiền xử lý ảnh."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image(gray, "grayscale.jpg")
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    save_image(blur, "blurred.jpg")
    gamma = adjust_gamma(blur, gamma=0.5)
    save_image(gamma, "gamma_adjusted.jpg")
    _, thresh = cv2.threshold(gamma, 0, 255,
                              cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    save_image(thresh, "thresholded.jpg")
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 5))
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)
    save_image(dilation, "dilated.jpg")
    return dilation


def extract_characters(dilation, rotated):
    """Trích xuất các ký tự từ ảnh biển số đã xử lý."""
    cnts, _ = cv2.findContours(dilation, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)
    plate_cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[1]
    cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

    height = cv2.boundingRect(plate_cnt)[-1]
    width = cv2.boundingRect(plate_cnt)[-2]
    top_half, bottom_half, rois = [], [], []

    for i, cnt in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(cnt)
        if height / h > 4 or height / h < 1.5:
            continue
        if h / w > 3:
            continue
        roi = dilation[y - 3:y + 3 + h, x - 3:x + 3 + w]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)

        if width / height >= 3.5:
            rois.append(roi)
        else:
            if y < height / 2 - 20:
                top_half.append(roi)
            else:
                bottom_half.append(roi)

    return top_half + bottom_half if (top_half and bottom_half) else rois


def recognize_plate_characters(rois):
    plate_num = ''
    for roi in rois:
        char = pytesseract.image_to_string(
            roi,
            config=
            '-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3'
        ).strip()
        plate_num += char
    return plate_num


def detect_license_plate(file):
    """Chuỗi xử lý chính để phát hiện và nhận dạng biển số xe."""
    plates = get_plate_predictions(file)
    if not plates:
        return "Không tìm thấy biển số"
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_plate = crop_plate(img, max(plates, key=lambda plate: plate['height']))
    rotated = deskew(resized_plate)
    processed = preprocess_image(rotated)
    rois = extract_characters(processed, rotated)
    max_height = max(roi.shape[0] for roi in rois)
    _rois = [cv2.resize(roi, (roi.shape[1], max_height)) for roi in rois]
    save_image(cv2.hconcat(_rois), "rois.jpg")
    plate_num = recognize_plate_characters(rois)
    return plate_num
