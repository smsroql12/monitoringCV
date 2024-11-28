#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <future>
#include <mutex>
#include "inference.h"
#include "deepsort.h"

using namespace std;
using namespace cv;

// ���� ������ mutex
Mat displayFrame;
mutex frameMutex;
bool stopProcessing = false;
double realFPS = 0.0;

// GrabCut ��� ���׸����̼� �Լ�
void applyGrabCut(const Mat& frame, Rect& bbox, Mat& outputMask) {
    // �ٿ�� �ڽ� ���� �� ����
    bbox.x = max(0, bbox.x);
    bbox.y = max(0, bbox.y);
    bbox.width = min(frame.cols - bbox.x, bbox.width);
    bbox.height = min(frame.rows - bbox.y, bbox.height);

    if (bbox.area() <= 0) {
        cerr << "Invalid bounding box for GrabCut: " << bbox << endl;
        outputMask = Mat(frame.size(), CV_8UC1, Scalar(0)); // �� ����ũ ��ȯ
        return;
    }

    Mat bgModel, fgModel;
    Mat mask(frame.size(), CV_8UC1, Scalar(GC_BGD));
    rectangle(mask, bbox, Scalar(GC_PR_FGD), -1);

    try {
        grabCut(frame, mask, bbox, bgModel, fgModel, 1, GC_INIT_WITH_RECT);
        compare(mask, GC_PR_FGD, outputMask, CMP_EQ); // ����ũ ���̳ʸ�ȭ
    }
    catch (const cv::Exception& e) {
        cerr << "GrabCut failed: " << e.what() << endl;
        outputMask = Mat(frame.size(), CV_8UC1, Scalar(0)); // �� ����ũ ��ȯ
    }
}

// �ð�ȭ �Լ�
void visualizeSegmentation(Mat& frame, const vector<DeepSORT::Track>& tracks) {
    Mat grayFrame;
    cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
    cvtColor(grayFrame, grayFrame, COLOR_GRAY2BGR);

    Mat resultFrame = grayFrame.clone();

    vector<future<void>> futures;
    for (const auto& track : tracks) {
        futures.emplace_back(async(launch::async, [&]() {
            Mat mask;
            Rect bbox = track.bbox;

            applyGrabCut(frame, bbox, mask);

            lock_guard<mutex> lock(frameMutex);
            if (!mask.empty() && countNonZero(mask) > bbox.area() * 0.1) {
                // Segmentation ����
                Mat objectFrame;
                frame.copyTo(objectFrame, mask);
                objectFrame.copyTo(resultFrame, mask);
            }
            else {
                // Segmentation ���� �� �ٿ�� �ڽ� ���� �÷� ó��
                frame(bbox).copyTo(resultFrame(bbox));
            }

            rectangle(resultFrame, bbox, Scalar(0, 255, 0), 1);
            putText(resultFrame, "ID: " + to_string(track.id),
                Point(bbox.x, bbox.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
            }));
    }

    // ��� ���� �۾� �Ϸ� ���
    for (auto& fut : futures) {
        try {
            fut.get();
        }
        catch (const exception& e) {
            cerr << "Async task failed: " << e.what() << endl;
        }
    }

    lock_guard<mutex> lock(frameMutex);
    displayFrame = resultFrame.clone();
}

// ������ ó�� �Լ�
void processFrame(Mat& frame, Inference& inf, DeepSORT& tracker, int targetWidth, int targetHeight, int frameSkip) {
    static int frameCount = 0;

    if (frameCount % frameSkip != 0) {
        frameCount++;
        return;
    }

    frameCount++;
    resize(frame, frame, Size(targetWidth, targetHeight));
    vector<Detection> detections = inf.runInference(frame);

    if (detections.empty()) {
        cerr << "No detections found in the frame" << endl;
        lock_guard<mutex> lock(frameMutex);
        displayFrame = frame.clone();
        return;
    }

    vector<DeepSORT::Track> tracks = tracker.update(detections, frame, frameCount);
    visualizeSegmentation(frame, tracks);
}

int main() {
    string modelPath = "C:\\Users\\Administrator\\source\\repos\\myOpenCV\\myOpenCV\\assets\\yolov8s.onnx";
    string videoPath = "C:\\Users\\Administrator\\source\\repos\\myOpenCV\\myOpenCV\\assets\\street.mp4";

    bool runOnGPU = true;
    Size resizeDim(640, 480);

    Inference inf(modelPath, resizeDim, "", runOnGPU);
    VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file" << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    cout << "Video FPS: " << fps << endl;
    int delay = static_cast<int>(1000 / fps);

    int frameSkip = 2; // ������ ��ŵ ���� (���� ����)

    DeepSORT tracker;

    thread processingThread([&]() {
        while (!stopProcessing) {
            Mat frame;
            cap >> frame;
            if (frame.empty()) {
                stopProcessing = true;
                break;
            }

            auto startTime = chrono::high_resolution_clock::now();
            processFrame(frame, inf, tracker, resizeDim.width, resizeDim.height, frameSkip);
            auto endTime = chrono::high_resolution_clock::now();

            realFPS = 1000.0 / chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
        }
        });

    while (!stopProcessing) {
        {
            lock_guard<mutex> lock(frameMutex);
            if (!displayFrame.empty()) {
                // FPS �� �ð�ȭ ���� ǥ��
                putText(displayFrame, "FPS: " + to_string(realFPS), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
                imshow("Processed Video with Segmentation", displayFrame);
            }
        }

        if (waitKey(delay) == 27) {
            stopProcessing = true;
            break;
        }
    }

    if (processingThread.joinable()) {
        processingThread.join();
    }

    //��ü ���� �� ����
    cap.release();
    destroyAllWindows();
    return 0;
}
