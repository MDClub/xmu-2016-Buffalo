#ifndef opencvtest
#define opencvtest

#include <opencv2/opencv.hpp>  
#include <cstdio>  
#include <cstdlib>  
#include <Windows.h>  
#include<numeric>
#include "math.h"

std::vector<cv::Rect> DetectAndMark(IplImage *pSrcImage,CvHaarClassifierCascade *pHaarClassCascade,int min_neighbor,CvSize minsize,bool templet);
std::vector<cv::Rect> boxmatch(IplImage *pSrcImage,cv::Size facewh,std::vector<cv::Rect> facebox,std::vector<float> featureVec0,float &tempchi);
void hMirrorTrans(const cv::Mat &src, cv::Mat &dst);
void gray01(cv::Mat &gray,std::vector<float> &featureVec);
void PCAconv(cv::Mat &src,cv::Mat &dst,cv::Mat &kernel,std::vector<float> &PCAfeature);
void PCA2fea(cv::Mat &src,cv::Mat &dst,std::vector<float> &PCAfeature);


#endif