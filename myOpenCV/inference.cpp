#include "inference.h"

Inference::Inference(const std::string& onnxModelPath, const cv::Size& modelInputShape, const std::string& classesTxtFile, const bool& runWithMPS)
{
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    classesPath = classesTxtFile;
    mpsEnabled = runWithMPS;

    loadOnnxNetwork();
}

std::vector<Detection> Inference::runInference(const cv::Mat& input)
{
    cv::Mat modelInput;
    if (letterBoxForSquare && modelShape.width == modelShape.height)
        modelInput = formatToSquare(input);
    else
        modelInput = input.clone();

    // Create input blob
    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
    net.setInput(blob);

    // Forward pass
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Parse YOLOv8 results
    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    if (dimensions > rows) {
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];
        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }

    float x_factor = input.cols / modelShape.width;
    float y_factor = input.rows / modelShape.height;

    std::vector<Detection> detections;
    for (int i = 0; i < rows; ++i) {
        const float* data = (float*)outputs[0].data + i * dimensions;
        float confidence = data[4];

        if (confidence >= modelConfidenceThreshold) {
            float x = data[0], y = data[1];
            float w = data[2], h = data[3];

            int left = static_cast<int>((x - w / 2) * x_factor);
            int top = static_cast<int>((y - h / 2) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            detections.push_back({ 0, "person", confidence, cv::Scalar(0, 255, 0), cv::Rect(left, top, width, height) });
        }
    }

    // Non-Maximum Suppression
    std::vector<int> nms_indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    for (const auto& det : detections) {
        boxes.push_back(det.box);
        confidences.push_back(det.confidence);
    }

    cv::dnn::NMSBoxes(boxes, confidences, modelConfidenceThreshold, modelNMSThreshold, nms_indices);

    std::vector<Detection> filteredDetections;
    for (const int idx : nms_indices)
        filteredDetections.push_back(detections[idx]);

    return filteredDetections;
}

void Inference::loadOnnxNetwork()
{
    net = cv::dnn::readNetFromONNX(modelPath);
    if (mpsEnabled) {
        std::cout << "Running on MPS (Metal Performance Shaders)" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // MPS 자동 활성화
    }
    else {
        std::cout << "Running on CPU" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

cv::Mat Inference::formatToSquare(const cv::Mat& source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);

    cv::Mat result = cv::Mat::zeros(_max, _max, source.type());
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}
