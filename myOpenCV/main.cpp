#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "inference.h"
#include "deepsort.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    bool runOnGPU = true;

    // Inference Ŭ���� �ʱ�ȭ
    Inference inf("C:\\Users\\Administrator\\source\\repos\\myOpenCV\\myOpenCV\\assets\\yolov8s.onnx", Size(640, 480), "classes.txt", runOnGPU);

    // ������ �ε�
    string videoPath = "C:\\Users\\Administrator\\source\\repos\\myOpenCV\\myOpenCV\\assets\\test2.mp4";
    VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file" << endl;
        return -1;
    }

    Size resizeDim(640, 480);

    // DeepSORT �ʱ�ȭ
    DeepSORT tracker;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        resize(frame, frame, resizeDim);

        // YOLO ���� ���� Ž�� ����
        vector<Detection> detections = inf.runInference(frame);

        // Ž�� ����� DeepSORT �Է� �������� ��ȯ
        vector<DeepSORT::Track> tracks = tracker.update(detections, frame);

        Mat mask = Mat::zeros(frame.size(), CV_8UC1);

        for (const auto& track : tracks) {
            int id = track.id;
            Rect currentRect = track.bbox;
            Scalar color = Scalar(0, 255, 0);

            rectangle(frame, currentRect, color, 2);
            putText(frame, "ID: " + to_string(id),
                Point(currentRect.x, currentRect.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

            rectangle(mask, currentRect, Scalar(255), FILLED);
        }

        Mat invertedMask;
        bitwise_not(mask, invertedMask);

        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        cvtColor(grayFrame, grayFrame, COLOR_GRAY2BGR);

        Mat resultFrame;
        bitwise_and(grayFrame, grayFrame, resultFrame, invertedMask);
        frame.copyTo(resultFrame, mask);

        imshow("Person Detection with Grayscale Background", resultFrame);

        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}