#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "inference.h" // Inference class ���

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
        cerr << "Error: Could not open video file " << videoPath << endl;
        return -1;
    }

    // Resize ũ�� ����
    Size resizeDim(640, 480);

    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        resize(frame, frame, resizeDim);

        // YOLO �߷� ����
        vector<Detection> output = inf.runInference(frame);

        // ��ü �������� �׷��̽����Ϸ� ��ȯ
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        cvtColor(grayFrame, grayFrame, COLOR_GRAY2BGR); // 3ä�η� ��ȯ

        // ����ũ �ʱ�ȭ (������ ũ��� ����)
        Mat mask = Mat::zeros(frame.size(), CV_8UC1);

        // �� ������ ��ü�� ���� �ٿ�� �ڽ� �׸���
        for (const auto& detection : output)
        {
            Rect box = detection.box;
            Scalar color = detection.color;

            // ����ũ�� �ڽ� ���θ� ���(255)���� ����
            rectangle(mask, box, Scalar(255), FILLED);

            // �����ӿ� �÷� �ٿ�� �ڽ��� Ŭ���� �̸� ǥ��
            rectangle(frame, box, color, 2);
            string classString = detection.className + " " + to_string(detection.confidence).substr(0, 4);
            putText(frame, classString, Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        }

        // ����ũ ���� (�ڽ� ���ΰ� ������, �ܺΰ� ���)
        Mat invertedMask;
        bitwise_not(mask, invertedMask);

        // �׷��̽����� �����ӿ� ����ũ ���� (�ٱ��ʸ� �׷��̽�����)
        Mat resultFrame;
        bitwise_and(grayFrame, grayFrame, resultFrame, invertedMask); // �ٱ����� �׷��̽�����
        frame.copyTo(resultFrame, mask); // �ڽ� ���δ� �÷� ����

        // ��� ���
        imshow("Result Frame", resultFrame);

        // ESC Ű �Է� �� ����
        if (waitKey(1) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
