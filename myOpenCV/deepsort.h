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
        cv::Rect bbox;
        float confidence;
    };

    DeepSORT();
    std::vector<Track> update(const std::vector<Detection>& detections, const cv::Mat& frame);

private:
    int nextId;
    std::unordered_map<int, cv::Rect> trackedObjects;

    static constexpr float IOU_THRESHOLD = 0.3; // IOU �Ӱ谪
    float calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2);
};

#endif // DEEPSORT_H
