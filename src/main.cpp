#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iostream>
#include <filesystem>
#include <vector>
#include <math.h>
#include <string>

// Mock class for InferenceHTTPClient
class InferenceHTTPClient {
public:
    std::vector<std::map<std::string, float>> infer(const std::string& file, const std::string& model_id) {
        // Dummy implementation - replace with actual inference logic
        return {
            { {"x", 100}, {"y", 200}, {"width", 300}, {"height", 150} } // Example prediction
            }
        };
    }
};

// Global Inference Client
InferenceHTTPClient CLIENT;

void save_image(const cv::Mat& image, const std::string& filename) {
    if (!std::filesystem::exists("processed_images")) {
        std::filesystem::create_directory("processed_images");
    }
    cv::imwrite("processed_images/" + filename, image);
}

std::vector<std::map<std::string, float>> get_plate_predictions(const std::string& file, const std::string& model_id = "plate-detect-bkwoo/1") {
    return CLIENT.infer(file, model_id);
}

cv::Mat crop_plate(const cv::Mat& img, const std::map<std::string, float>& plate) {
    int x_0 = plate.at("x");
    int y_0 = plate.at("y");
    int w = plate.at("width");
    int h = plate.at("height");

    int x1 = static_cast<int>(x_0 - w / 2);
    int y1 = static_cast<int>(y_0 - h / 2);
    int x2 = static_cast<int>(x1 + w);
    int y2 = static_cast<int>(y1 + h);

    cv::Rect roi(x1 - 5, y1, (x2 + 5) - (x1 - 5), (y2 - y1));
    cv::Mat crop = img(roi);
    cv::Mat resize;
    cv::resize(crop, resize, cv::Size(), 5, 3, cv::INTER_CUBIC);
    save_image(resize, "resized_plate.jpg");
    return resize;
}

float compute_skew(const cv::Mat& image) {
    int h = image.rows;
    int w = image.cols;

    cv::Mat img;
    cv::medianBlur(image, img, 3);

    cv::Mat edges;
    cv::Canny(img, edges, 30, 100, 3, true);
    
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 30, w / 4.0, h / 4.0);

    double angle = 0.0;
    int cnt = 0;

    for (const auto& line : lines) {
        double ang = atan2(line[3] - line[1], line[2] - line[0]);
        if (fabs(ang) <= 30) {
            angle += ang;
            cnt++;
        }
    }

    if (cnt == 0) {
        return 0.0;
    }

    return (angle / cnt) * 180 / CV_PI;
}

cv::Mat deskew(const cv::Mat& image) {
    float angle = compute_skew(image);
    cv::Point2f center(static_cast<float>(image.cols) / 2, static_cast<float>(image.rows) / 2);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat deskewed;
    cv::warpAffine(image, deskewed, rot_mat, image.size(), cv::INTER_LINEAR);
    save_image(deskewed, "deskewed_plate.jpg");
    return deskewed;
}

cv::Mat adjust_gamma(const cv::Mat& image, double gamma = 1.0) {
    cv::Mat adjusted;
    double inv_gamma = 1.0 / gamma;
    cv::Mat lookup_table(1, 256, CV_8U);
    uchar* ptr = lookup_table.ptr();
    for (int i = 0; i < 256; ++i) {
        ptr[i] = static_cast<uchar>(pow(i / 255.0, inv_gamma) * 255);
    }
    cv::LUT(image, lookup_table, adjusted);
    return adjusted;
}

cv::Mat preprocess_image(const cv::Mat& image) {
    cv::Mat gray, blur, thresh, dilation;
    
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    save_image(gray, "grayscale.jpg");
    
    cv::GaussianBlur(gray, blur, cv::Size(9, 9), 0);
    save_image(blur, "blurred.jpg");
    
    cv::Mat gamma = adjust_gamma(blur, 0.5);
    save_image(gamma, "gamma_adjusted.jpg");
    
    cv::threshold(gamma, thresh, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    save_image(thresh, "thresholded.jpg");
    
    cv::Mat rect_kern = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 5));
    cv::dilate(thresh, dilation, rect_kern, cv::Point(-1, -1), 1);
    save_image(dilation, "dilated.jpg");
    
    return dilation;
}

std::vector<cv::Mat> extract_characters(const cv::Mat& dilation, const cv::Mat& rotated) {
    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(dilation, cnts, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Mat> rois;

    for (const auto& cnt : cnts) {
        cv::Rect roi = cv::boundingRect(cnt);
        if (roi.height < 10 || roi.width < 10) continue; // Filter out small contours
        cv::Mat char_roi = dilation(roi);
        cv::Mat inverted_roi;
        cv::bitwise_not(char_roi, inverted_roi);
        rois.push_back(inverted_roi);
    }


    return rois;
}

std::string recognize_plate_characters(const std::vector<cv::Mat>& rois) {
    std::string plate_num;
    tesseract::TessBaseAPI tess;

    if (tess.Init(nullptr, "eng")) {
        std::cerr << "Could not initialize tesseract." << std::endl;
        return "";
    }

    for (const auto& roi : rois) {
        tess.SetImage(roi.data, roi.cols, roi.rows, 1, roi.step);
        char* char_result = tess.GetUTF8Text();
        plate_num += char_result;
        delete[] char_result;
    }

    tess.End();
    return plate_num;
}

std::string detect_license_plate(const std::string& file) {
    auto plates = get_plate_predictions(file);
    if (plates.empty()) {
        return "Không tìm thấy biển số";
    }
    
    cv::Mat img = cv::imread(file);
    cv::Mat resized_plate = crop_plate(img, plates[0]);
    cv::Mat rotated = deskew(resized_plate);
    cv::Mat processed = preprocess_image(rotated);
    std::vector<cv::Mat> rois = extract_characters(processed, rotated);
    
    if (rois.empty()) {
        return "Không tìm thấy ký tự";
    }
    
    std::string plate_num = recognize_plate_characters(rois);
    return plate_num;
}

int main() {
    std::string file = "path/to/your/image.jpg"; // Update with your image path
    std::string plate_number = detect_license_plate(file);
    std::cout << "Plate Number: " << plate_number << std::endl;
    return 0;
}
