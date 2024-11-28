#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include "inference.h"

class DeepSORT {
public:
    struct Track {
        int id;
        cv::Rect bbox;         // 현재 객체 바운딩 박스
        float confidence;      // 신뢰도 점수
        int age;               // 객체의 나이 (업데이트된 프레임 수)
        int lastUpdate;        // 마지막 업데이트 프레임 번호
        cv::Point2f velocity;  // 객체의 속도
        float reliability;     // 신뢰도 점수
    };

    DeepSORT();
    std::vector<Track> update(const std::vector<Detection>& detections, const cv::Mat& frame, int frameNumber);

private:
    int nextId; // 새로운 객체 ID
    std::unordered_map<int, Track> trackedObjects; // 현재 추적 중인 객체

    static constexpr float IOU_THRESHOLD = 0.3; // IOU 임계값 (객체 매칭 임계값)
    static constexpr float RELIABILITY_THRESHOLD = 0.5; // 최소 신뢰도
    static constexpr int MAX_AGE = 30; // 객체 최대 수명 (프레임 수 기준)
    static constexpr float CONFIDENCE_THRESHOLD = 0.5; // 최소 confidence 값
    static constexpr int SMOOTHING_WINDOW = 5; // 속도 계산 시 스무딩을 위한 프레임 수

    float calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2);
    cv::Point2f calculateVelocity(const cv::Rect& prevRect, const cv::Rect& currentRect);
    void updateReliability(Track& track, bool matched);
    void removeUnreliableTracks();
    void removeStaleTracks(int frameNumber);
    void smoothTrackMovement(Track& track, const cv::Point2f& newVelocity); // 이동 평활화
};

#endif // DEEPSORT_H
