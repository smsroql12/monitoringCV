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

    // Resize 크기 정의
    Size resizeDim(640, 480);

    while (true)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        resize(frame, frame, resizeDim);

        // YOLO 추론 실행
        vector<Detection> output = inf.runInference(frame);

        // 전체 프레임을 그레이스케일로 변환
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        cvtColor(grayFrame, grayFrame, COLOR_GRAY2BGR); // 3채널로 변환

        // 마스크 초기화 (프레임 크기와 동일)
        Mat mask = Mat::zeros(frame.size(), CV_8UC1);

        // 각 감지된 객체에 대해 바운딩 박스 그리기
        for (const auto& detection : output)
        {
            Rect box = detection.box;
            Scalar color = detection.color;

            // 마스크에 박스 내부를 흰색(255)으로 설정
            rectangle(mask, box, Scalar(255), FILLED);

            // 프레임에 컬러 바운딩 박스와 클래스 이름 표시
            rectangle(frame, box, color, 2);
            string classString = detection.className + " " + to_string(detection.confidence).substr(0, 4);
            putText(frame, classString, Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        }

        // 마스크 반전 (박스 내부가 검은색, 외부가 흰색)
        Mat invertedMask;
        bitwise_not(mask, invertedMask);

        // 그레이스케일 프레임에 마스크 적용 (바깥쪽만 그레이스케일)
        Mat resultFrame;
        bitwise_and(grayFrame, grayFrame, resultFrame, invertedMask); // 바깥쪽은 그레이스케일
        frame.copyTo(resultFrame, mask); // 박스 내부는 컬러 유지

        // 결과 출력
        imshow("Result Frame", resultFrame);

        // ESC 키 입력 시 종료
        if (waitKey(1) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
