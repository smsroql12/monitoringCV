#include "deepsort.h"

DeepSORT::DeepSORT() : nextId(0) {}

std::vector<DeepSORT::Track> DeepSORT::update(const std::vector<Detection>& detections, const cv::Mat& frame, int frameNumber) {
    std::vector<Track> tracks;
    std::unordered_map<int, Track> newTrackedObjects;

    for (const auto& detection : detections) {
        if (detection.confidence < CONFIDENCE_THRESHOLD) continue; // 낮은 confidence 필터링

        cv::Rect currentRect = detection.box;
        int matchedId = -1;
        float bestIOU = 0.0;

        // 기존 객체와 IOU 계산
        for (const auto& [id, track] : trackedObjects) {
            float iou = calculateIOU(track.bbox, currentRect);
            if (iou > bestIOU && iou > IOU_THRESHOLD) {
                matchedId = id;
                bestIOU = iou;
            }
        }

        // 새 객체라면 새로운 ID 부여
        if (matchedId == -1) {
            matchedId = nextId++;
        }

        // 속도 계산 (기존 객체가 있을 경우)
        cv::Point2f velocity = cv::Point2f(0.0, 0.0);
        if (trackedObjects.find(matchedId) != trackedObjects.end()) {
            velocity = calculateVelocity(trackedObjects[matchedId].bbox, currentRect);
        }

        // 새로운 객체 업데이트
        Track newTrack = { matchedId, currentRect, detection.confidence, 0, frameNumber, velocity };
        newTrackedObjects[matchedId] = newTrack;

        // 추적 결과 저장
        tracks.push_back(newTrack);
    }

    // 기존 객체의 나이 업데이트 및 유지
    for (auto& [id, track] : trackedObjects) {
        if (newTrackedObjects.find(id) == newTrackedObjects.end()) {
            track.age++;
            if (track.age < MAX_AGE) {
                // 위치 예측 (기존 속도 사용)
                track.bbox.x += static_cast<int>(track.velocity.x);
                track.bbox.y += static_cast<int>(track.velocity.y);
                newTrackedObjects[id] = track;
                tracks.push_back(track);
            }
        }
    }

    // 오래된 객체 제거
    removeStaleTracks(frameNumber);

    // 업데이트된 객체 저장
    trackedObjects = std::move(newTrackedObjects);

    return tracks;
}

// 오래된 객체 제거
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

// IOU 계산 함수
float DeepSORT::calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2) {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = rect1.area() + rect2.area() - intersectionArea;
    return static_cast<float>(intersectionArea) / unionArea;
}

// 속도 계산 함수
cv::Point2f DeepSORT::calculateVelocity(const cv::Rect& prevRect, const cv::Rect& currentRect) {
    return cv::Point2f(currentRect.x - prevRect.x, currentRect.y - prevRect.y);
}
