#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "inference.h" // Inference class 사용

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    bool runOnGPU = true;

    // Inference 클래스 초기화
    Inference inf("C:\\Users\\Administrator\\source\\repos\\myOpenCV\\myOpenCV\\assets\\yolov8s.onnx", Size(640, 480), "classes.txt", runOnGPU);

    // 동영상 로드
    string videoPath = "C:\\Users\\Administrator\\source\\repos\\myOpenCV\\myOpenCV\\assets\\test2.mp4";
    VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file " << videoPath << endl;
        return -1;
    }

    Size resizeDim(640, 480);

    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        resize(frame, frame, resizeDim);

        vector<Detection> output = inf.runInference(frame);

        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        cvtColor(grayFrame, grayFrame, COLOR_GRAY2BGR);

        Mat mask = Mat::zeros(frame.size(), CV_8UC1);

        for (const auto& detection : output)
        {
            Rect box = detection.box;
            Scalar color = detection.color;

            rectangle(mask, box, Scalar(255), FILLED);
            rectangle(frame, box, color, 2);

            // 신뢰도 점수와 함께 "Person" 표시
            string classString = "Person " + to_string(detection.confidence).substr(0, 4);
            putText(frame, classString, Point(box.x, box.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        }

        Mat invertedMask;
        bitwise_not(mask, invertedMask);

        Mat resultFrame;
        bitwise_and(grayFrame, grayFrame, resultFrame, invertedMask);
        frame.copyTo(resultFrame, mask);

        imshow("Person Detection", resultFrame);

        if (waitKey(1) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
