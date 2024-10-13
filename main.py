import cv2
from inference_sdk import InferenceHTTPClient
import pytesseract

file = 'tesssssst.jpg'
CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com",
                             api_key="8PKnBAYVuzxib2KznYlV")


def detect_plate(file, model_id="plate-detect-bkwoo/1"):
    result = CLIENT.infer(file, model_id=model_id)
    plates = result.get('predictions', [])
    return plates


def draw_bounding_box(img, plate):
    x_0, y_0, w, h = plate['x'], plate['y'], plate['width'], plate['height']
    x_1, y_1 = x_0 - w / 2, y_0 - h / 2
    x_2, y_2 = x_1 + w, y_1 + h
    cv2.rectangle(img, (int(x_1) + 5, int(y_1)), (int(x_2) - 5, int(y_2)),
                  (0, 255, 0), 1)
    return img, (int(x_1), int(y_1), int(x_2), int(y_2))


def crop_plate(img, coordinates):
    x1, y1, x2, y2 = coordinates
    crop = img[y1:y2, x1 + 5:x2 - 5]
    return crop


def preprocess_image(crop):
    resize = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resize, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return resize, thresh


def extract_characters(thresh, resize):
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours,
                             key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    height, _, _ = resize.shape
    rois = []
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if height / h > 4 or height / h < 1.5:
            continue
        roi = thresh[y - 5:y + h + 5, x - 5:x + w + 5]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)
        rois.append(roi)

    #     if width / height >= 3.5:
    #         chars.append(char)
    #     else:
    #         if y < height / 2 - 20:
    #             top_half.append(char)
    #         else:
    #             bottom_half.append(char)
    # return ''.join(chars) if not (
    #     top_half and bottom_half) else ''.join(top_half) + ''.join(bottom_half)
    return roi

import os
def main():
    for filename in os.listdir("data"):
        plate = cv2.imread(f"data/{filename}")
        resize, thresh = preprocess_image(plate)
        output = extract_characters(resize, thresh)
        for i in range(len(output)): 
            cv2.imwrite(f"charSegments/{filename}_{i}.png", output[i]) 
main()