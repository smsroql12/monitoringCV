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
        cv::Rect bbox;         // ���� ��ü �ٿ�� �ڽ�
        float confidence;      // �ŷڵ� ����
        int age;               // ��ü�� ���� (������Ʈ�� ������ ��)
        int lastUpdate;        // ������ ������Ʈ ������ ��ȣ
        cv::Point2f velocity;  // ��ü�� �ӵ�
        float reliability;     // �ŷڵ� ����
    };

    DeepSORT();
    std::vector<Track> update(const std::vector<Detection>& detections, const cv::Mat& frame, int frameNumber);

private:
    int nextId; // ���ο� ��ü ID
    std::unordered_map<int, Track> trackedObjects; // ���� ���� ���� ��ü

    static constexpr float IOU_THRESHOLD = 0.3; // IOU �Ӱ谪 (��ü ��Ī �Ӱ谪)
    static constexpr float RELIABILITY_THRESHOLD = 0.5; // �ּ� �ŷڵ�
    static constexpr int MAX_AGE = 30; // ��ü �ִ� ���� (������ �� ����)
    static constexpr float CONFIDENCE_THRESHOLD = 0.5; // �ּ� confidence ��
    static constexpr int SMOOTHING_WINDOW = 5; // �ӵ� ��� �� �������� ���� ������ ��

    float calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2);
    cv::Point2f calculateVelocity(const cv::Rect& prevRect, const cv::Rect& currentRect);
    void updateReliability(Track& track, bool matched);
    void removeUnreliableTracks();
    void removeStaleTracks(int frameNumber);
    void smoothTrackMovement(Track& track, const cv::Point2f& newVelocity); // �̵� ��Ȱȭ
};

#endif // DEEPSORT_H
