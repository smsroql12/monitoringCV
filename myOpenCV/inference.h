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
    int class_id{ 0 };                // 감지된 객체 클래스 ID
    std::string className{};        // 클래스 이름
    float confidence{ 0.0 };          // 신뢰도 점수
    cv::Scalar color{};             // 바운딩 박스 색상
    cv::Rect box{};                 // 바운딩 박스 좌표
    cv::Mat mask{};                 // 세그멘테이션 마스크 (선택적)
};

class Inference {
public:
    Inference(const std::string& onnxModelPath, const cv::Size& modelInputShape = { 640, 640 }, const std::string& classesTxtFile = "", const bool& runWithMPS = true);

    // 객체 감지 실행
    std::vector<Detection> runInference(const cv::Mat& input);

    // 세그멘테이션 출력 처리
    std::vector<cv::Mat> extractSegmentationMasks(const cv::Mat& output, int rows, int cols);

private:
    void loadClassesFromFile();       // 클래스 파일 로드
    void loadOnnxNetwork();           // ONNX 모델 로드
    cv::Mat formatToSquare(const cv::Mat& source); // 입력 이미지 변환

    std::string modelPath{};          // ONNX 모델 경로
    std::string classesPath{};        // 클래스 파일 경로
    bool mpsEnabled{};                // MPS 사용 여부

    std::vector<std::string> classes{ "person" }; // 기본 클래스

    cv::Size2f modelShape{};          // 모델 입력 크기

    float modelConfidenceThreshold{ 0.25f }; // Confidence threshold
    float modelScoreThreshold{ 0.45f };      // Score threshold
    float modelNMSThreshold{ 0.50f };        // Non-maximum suppression threshold

    bool letterBoxForSquare{ true };    // 입력 이미지를 정사각형으로 변환할지 여부

    cv::dnn::Net net;                 // ONNX 모델을 로드하는 네트워크
};

#endif // INFERENCE_H
