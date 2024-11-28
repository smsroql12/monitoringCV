#ifndef INFERENCE_H
#define INFERENCE_H

#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct Detection {
    int class_id{ 0 };                // ������ ��ü Ŭ���� ID
    std::string className{};        // Ŭ���� �̸�
    float confidence{ 0.0 };          // �ŷڵ� ����
    cv::Scalar color{};             // �ٿ�� �ڽ� ����
    cv::Rect box{};                 // �ٿ�� �ڽ� ��ǥ
    cv::Mat mask{};                 // ���׸����̼� ����ũ (������)
};

class Inference {
public:
    Inference(const std::string& onnxModelPath, const cv::Size& modelInputShape = { 640, 640 }, const std::string& classesTxtFile = "", const bool& runWithMPS = true);

    // ��ü ���� ����
    std::vector<Detection> runInference(const cv::Mat& input);

    // ���׸����̼� ��� ó��
    std::vector<cv::Mat> extractSegmentationMasks(const cv::Mat& output, int rows, int cols);

private:
    void loadClassesFromFile();       // Ŭ���� ���� �ε�
    void loadOnnxNetwork();           // ONNX �� �ε�
    cv::Mat formatToSquare(const cv::Mat& source); // �Է� �̹��� ��ȯ

    std::string modelPath{};          // ONNX �� ���
    std::string classesPath{};        // Ŭ���� ���� ���
    bool mpsEnabled{};                // MPS ��� ����

    std::vector<std::string> classes{ "person" }; // �⺻ Ŭ����

    cv::Size2f modelShape{};          // �� �Է� ũ��

    float modelConfidenceThreshold{ 0.25f }; // Confidence threshold
    float modelScoreThreshold{ 0.45f };      // Score threshold
    float modelNMSThreshold{ 0.50f };        // Non-maximum suppression threshold

    bool letterBoxForSquare{ true };    // �Է� �̹����� ���簢������ ��ȯ���� ����

    cv::dnn::Net net;                 // ONNX ���� �ε��ϴ� ��Ʈ��ũ
};

#endif // INFERENCE_H
