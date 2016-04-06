#include "opencvtest.h"

using namespace std;  
using namespace cv; 

//分块求特征map的直方图
/*
gray：输入由PCA2fea函数得到的叠加后最终的feature map
featureVec：输出特征map分块统计的直方图特征向量
*/
void gray01(Mat &gray,vector<float> &featureVec)
{
	int channels[] = {0};//第一个通道
	int histSize[] = {8};//直方图分为几份
	float granges [] = {0.0,7.0};//灰度图取值范围
	const float* ranges[] = {granges};

	int blocksize = 10;//分块大小

	double minval=0;
	double maxval=0;

	float temp=0.0;

	for(int i=0;i<gray.rows;)
	{
		for(int j=0;j<gray.cols;)
		{
			//对每个块内的每一列求直方图并拉成一个列向量
			Rect roi(j, i, 1, blocksize);
			Mat roi_img = gray(roi);
			Mat hist;//直方图

			calcHist(&roi_img,1,channels,Mat(),hist,1,histSize,ranges,true,false);

			//minMaxLoc(hist, 0, &maxval, 0, 0);//求最大值

			for(int k=0;k<hist.rows;k++)
			{	
				temp = hist.at<float>(k,0);
				featureVec.push_back(temp);
			}
			j+=1;
		}
		i+=blocksize;
	}
}