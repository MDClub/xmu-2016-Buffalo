#include "opencvtest.h"

using namespace std;  
using namespace cv; 

extern Mat kernel1;
extern Mat kernel2;
extern Mat kernel3;


//图像与PCAnet训练得到的卷积核进行卷积
/*
src：输入灰度图像
dst：输出经过卷积后的二值图像
kernel：输入卷积核
PCAfeature：将二值图像拉成一行向量输出
*/
void PCAconv(Mat &src,Mat &dst,Mat &kernel,vector<float> &PCAfeature)
{
	if(src.depth() != CV_32F) 
		src.convertTo(src,CV_32F,1/255.);//转为0~1
	
	float temp = 0.0;

	filter2D(src,dst,src.depth(),kernel);

	for(int i=0;i<dst.rows;i++)
	{
		for(int j=0;j<dst.cols;j++)
		{
			//二值化，将大于0的置为1
			temp = dst.at<float>(i,j);
			//temp = abs(temp);
			if(temp>0.001)
			{
				PCAfeature.push_back(1.0);
				dst.at<float>(i,j) = 1.0;
			}
			else
			{
				PCAfeature.push_back(0.0);
				dst.at<float>(i,j) = 0.0;
			}			
		}
	}
}


//将经过三个卷积和卷积后的三张feature map加权相加，合成一张feature map，并拉成一个特征向量
/*
src：输入灰度图像
dst：输出叠加后最终的feature map
PCAfeature：输出特征向量
*/
void PCA2fea(Mat &src,Mat &dst,vector<float> &PCAfeature)
{
	vector<float> PCAfeature1;
	vector<float> PCAfeature2;
	vector<float> PCAfeature3;
	Mat dst1;
	Mat dst2;
	Mat dst3;

	PCAconv(src,dst1,kernel1,PCAfeature1);
	PCAconv(src,dst2,kernel2,PCAfeature2);
	PCAconv(src,dst3,kernel3,PCAfeature3);
	addWeighted(dst1, 1.0, dst2, 2.0, 0.0, dst);
	addWeighted(dst, 1.0, dst3, 4.0, 0.0, dst);

	for(int i=0;i<dst.rows;i++)
	{
		for(int j=0;j<dst.cols;j++)
		{
			PCAfeature.push_back(dst.at<float>(i,j));
		}
	}

}