#include "deepsort.h"

DeepSORT::DeepSORT() : nextId(0) {}

std::vector<DeepSORT::Track> DeepSORT::update(const std::vector<Detection>& detections, const cv::Mat& frame) {
    std::vector<Track> tracks;

    std::unordered_map<int, cv::Rect> newTrackedObjects;

    for (const auto& detection : detections) {
        cv::Rect currentRect = detection.box;
        int matchedId = -1;
        float bestIOU = 0.0;

        // ���� ��ü�� IOU ���
        for (const auto& [id, rect] : trackedObjects) {
            float iou = calculateIOU(rect, currentRect);
            if (iou > bestIOU && iou > IOU_THRESHOLD) { // IOU_THRESHOLD = 0.3
                matchedId = id;
                bestIOU = iou;
            }
        }

        // �� ��ü��� ���ο� ID �ο�
        if (matchedId == -1) {
            matchedId = nextId++;
        }

        // ���ο� ��ü ������Ʈ
        newTrackedObjects[matchedId] = currentRect;

        // ���� ��� ����
        tracks.push_back({ matchedId, currentRect, detection.confidence });
    }

    // ������ ��ü ���� (��ȿ�� �˻�)
    trackedObjects.clear();
    trackedObjects = newTrackedObjects;

    return tracks;
}

// IOU ��� �Լ�
float DeepSORT::calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2) {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = rect1.area() + rect2.area() - intersectionArea;
    return static_cast<float>(intersectionArea) / unionArea;
}
