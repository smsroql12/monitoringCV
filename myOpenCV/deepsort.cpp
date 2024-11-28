#include "deepsort.h"

DeepSORT::DeepSORT() : nextId(0) {}

std::vector<DeepSORT::Track> DeepSORT::update(const std::vector<Detection>& detections, const cv::Mat& frame, int frameNumber) {
    std::vector<Track> tracks;
    std::unordered_map<int, Track> newTrackedObjects;

    for (const auto& detection : detections) {
        if (detection.confidence < CONFIDENCE_THRESHOLD) continue; // ���� confidence ���͸�

        cv::Rect currentRect = detection.box;
        int matchedId = -1;
        float bestIOU = 0.0;

        // ���� ��ü�� IOU ���
        for (const auto& [id, track] : trackedObjects) {
            float iou = calculateIOU(track.bbox, currentRect);
            if (iou > bestIOU && iou > IOU_THRESHOLD) {
                matchedId = id;
                bestIOU = iou;
            }
        }

        // �� ��ü��� ���ο� ID �ο�
        if (matchedId == -1) {
            matchedId = nextId++;
        }

        // �ӵ� ��� (���� ��ü�� ���� ���)
        cv::Point2f velocity = cv::Point2f(0.0, 0.0);
        if (trackedObjects.find(matchedId) != trackedObjects.end()) {
            velocity = calculateVelocity(trackedObjects[matchedId].bbox, currentRect);
        }

        // ���ο� ��ü ������Ʈ
        Track newTrack = { matchedId, currentRect, detection.confidence, 0, frameNumber, velocity };
        newTrackedObjects[matchedId] = newTrack;

        // ���� ��� ����
        tracks.push_back(newTrack);
    }

    // ���� ��ü�� ���� ������Ʈ �� ����
    for (auto& [id, track] : trackedObjects) {
        if (newTrackedObjects.find(id) == newTrackedObjects.end()) {
            track.age++;
            if (track.age < MAX_AGE) {
                // ��ġ ���� (���� �ӵ� ���)
                track.bbox.x += static_cast<int>(track.velocity.x);
                track.bbox.y += static_cast<int>(track.velocity.y);
                newTrackedObjects[id] = track;
                tracks.push_back(track);
            }
        }
    }

    // ������ ��ü ����
    removeStaleTracks(frameNumber);

    // ������Ʈ�� ��ü ����
    trackedObjects = std::move(newTrackedObjects);

    return tracks;
}

// ������ ��ü ����
void DeepSORT::removeStaleTracks(int frameNumber) {
    for (auto it = trackedObjects.begin(); it != trackedObjects.end(); ) {
        if (frameNumber - it->second.lastUpdate > MAX_AGE) {
            it = trackedObjects.erase(it);
        }
        else {
            ++it;
        }
    }
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

// �ӵ� ��� �Լ�
cv::Point2f DeepSORT::calculateVelocity(const cv::Rect& prevRect, const cv::Rect& currentRect) {
    return cv::Point2f(currentRect.x - prevRect.x, currentRect.y - prevRect.y);
}
