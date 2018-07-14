
#include <stdio.h>
#include<thread>

#include <opencv2/opencv.hpp>
#include "../include/facedetect-dll.h"

//#pragma comment(lib,"libfacedetect.lib")
#pragma comment(lib,"../lib/libfacedetect-x64.lib")

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
using namespace cv;

//只检测正面脸和68个人脸关键点，不能检测侧面脸
void frontal(cv::Mat &srcImg, unsigned char * pBuffer, int * pResults, int &doLandmark)
{
	Mat gray;
	cvtColor(srcImg, gray, CV_BGR2GRAY);
	///////////////////////////////////////////
	// frontal face detection / 68 landmark detection
	// it's fast, but cannot detect side view faces
	//////////////////////////////////////////
	//!!! The input image must be a gray one (single-channel)
	//!!! DO NOT RELEASE pResults !!!
	pResults = facedetect_frontal(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
		1.2f, 2, 48, 0, doLandmark);

	//printf("%d faces detected.\n", (pResults ? *pResults : 0));
	//print the detection results
	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults + 1)) + 142 * i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int neighbors = p[4];
		int angle = p[5];

		printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
		rectangle(srcImg, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
		if (doLandmark)
		{
			for (int j = 0; j < 68; j++)
				circle(srcImg, Point((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]), 1, Scalar(0, 255, 0));
		}
	}
	imshow("Results_frontal", srcImg);
	waitKey(5);
}

//适合安防场景,对光照适应性较强,检测正脸和68个关键点
void frontal_surveillance(cv::Mat &srcImg, unsigned char * pBuffer, int * pResults, int &doLandmark)
{
	cv::Mat gray;
	cvtColor(srcImg, gray, CV_BGR2GRAY);
	///////////////////////////////////////////
	// frontal face detection designed for video surveillance / 68 landmark detection
	// it can detect faces with bad illumination.
	//////////////////////////////////////////
	//!!! The input image must be a gray one (single-channel)
	//!!! DO NOT RELEASE pResults !!!
	pResults = facedetect_frontal_surveillance(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
		1.2f, 2, 48, 0, doLandmark);
	//printf("%d faces detected.\n", (pResults ? *pResults : 0));
	
	//print the detection results
	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults + 1)) + 142 * i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int neighbors = p[4];
		int angle = p[5];

		printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
		rectangle(srcImg, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
		if (doLandmark)
		{
			for (int j = 0; j < 68; j++)
				circle(srcImg, Point((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]), 1, Scalar(0, 255, 0));
		}
	}
	imshow("Results_frontal_surveillance", srcImg);
	waitKey(5);
}

//多视角人脸检测，68个人脸关键点
void multiview(cv::Mat &srcImg, unsigned char * pBuffer, int * pResults, int &doLandmark)
{
	cv::Mat gray;
	cvtColor(srcImg, gray, CV_BGR2GRAY);
	///////////////////////////////////////////
	// multiview face detection / 68 landmark detection
	// it can detect side view faces, but slower than facedetect_frontal().
	//////////////////////////////////////////
	//!!! The input image must be a gray one (single-channel)
	//!!! DO NOT RELEASE pResults !!!
	pResults = facedetect_multiview(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
		1.2f, 2, 48, 0, doLandmark);

	//printf("%d faces detected.\n", (pResults ? *pResults : 0));
	//print the detection results
	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults + 1)) + 142 * i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int neighbors = p[4];
		int angle = p[5];

		printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
		rectangle(srcImg, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
		if (doLandmark)
		{
			for (int j = 0; j < 68; j++)
				circle(srcImg, Point((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]), 1, Scalar(0, 255, 0));
		}
	}
	imshow("Results_multiview", srcImg);
	waitKey(5);
}

//增强型多视角人脸检测，68个人脸关键点
void multiview_reinforce(Mat &srcImg, unsigned char * pBuffer, int * pResults, int &doLandmark)
{
	Mat gray;
	cvtColor(srcImg, gray, CV_BGR2GRAY);

	///////////////////////////////////////////
	// reinforced multiview face detection / 68 landmark detection
	// it can detect side view faces, better but slower than facedetect_multiview().
	//////////////////////////////////////////
	//!!! The input image must be a gray one (single-channel)
	//!!! DO NOT RELEASE pResults !!!
	pResults = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
		1.2f, 3, 48, 0, doLandmark);

	//printf("%d faces detected.\n", (pResults ? *pResults : 0));
	//print the detection results
	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults + 1)) + 142 * i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int neighbors = p[4];
		int angle = p[5];

		printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
		rectangle(srcImg, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
		if (doLandmark)
		{
			for (int j = 0; j < 68; j++)
				circle(srcImg, Point((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]), 1, Scalar(0, 255, 0));
		}
	}
	imshow("Results_multiview_reinforce", srcImg);
	waitKey(5);
}

//线程函数
void frontalThreadFunc(cv::VideoCapture  &Capture)
{
	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer)
	{
		fprintf(stderr, "Can not alloc buffer.\n");
		return;
	}
	int doLandmark = 1;

	cv::Mat Frame;
	while (Capture.isOpened())
	{
		int * pResults = NULL;
		Capture >> Frame;
		if (!Frame.empty())
		{
			frontal(Frame, pBuffer, pResults, doLandmark);
		}
	}
	free(pBuffer);
}

void frontal_surveillanceThreadFunc(cv::VideoCapture  &Capture)
{
	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer)
	{
		fprintf(stderr, "Can not alloc buffer.\n");
		return;
	}
	int doLandmark = 1;

	cv::Mat Frame;
	while (Capture.isOpened())
	{
		int * pResults = NULL;
		Capture >> Frame;
		if (!Frame.empty())
		{
			frontal_surveillance(Frame, pBuffer, pResults, doLandmark);
		}
	}
	free(pBuffer);
}

void multiviewThreadFunc(cv::VideoCapture  &Capture)
{
	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer)
	{
		fprintf(stderr, "Can not alloc buffer.\n");
		return;
	}
	int doLandmark = 1;

	cv::Mat Frame;
	cv::Mat frontalFrame;
	cv::Mat frontal_surveillanceFrame;
	cv::Mat multiviewFrame;
	cv::Mat multiview_reinforceFrame;
	while (Capture.isOpened())
	{
		int * pResults = NULL;
		Capture >> Frame;
		if (!Frame.empty())
		{
			multiview(Frame, pBuffer, pResults, doLandmark);
		}
	}
	free(pBuffer);
}

void multiview_reinforceThreadFunc(cv::VideoCapture  &Capture)
{
	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer)
	{
		fprintf(stderr, "Can not alloc buffer.\n");
		return;
	}
	int doLandmark = 1;

	cv::Mat Frame;
	while (Capture.isOpened())
	{
		int * pResults = NULL;
		Capture >> Frame;
		if (!Frame.empty())
		{
			multiview_reinforce(Frame, pBuffer, pResults, doLandmark);
		}
	}
	free(pBuffer);
}

int main()
{
	cv::VideoCapture  Capture;
	Capture.open(0);//打开第0个摄像头

	//创建4个线程，调用同一个摄像头同时使用四种人脸检测算法，分别进行人脸检测
	std::thread frontalThread(frontalThreadFunc, std::ref(Capture));	
	std::thread frontal_surveillanceThread(frontal_surveillanceThreadFunc, std::ref(Capture));
	std::thread multiviewThread(multiviewThreadFunc, std::ref(Capture));
	std::thread multiview_reinforceThread(multiview_reinforceThreadFunc, std::ref(Capture));

	frontalThread.join();
	frontal_surveillanceThread.join();
	multiviewThread.join();
	multiview_reinforceThread.join();

	return 0;
}