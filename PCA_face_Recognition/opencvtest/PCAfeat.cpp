#include "opencvtest.h"

using namespace std;  
using namespace cv; 

extern Mat kernel1;
extern Mat kernel2;
extern Mat kernel3;


//ͼ����PCAnetѵ���õ��ľ���˽��о��
/*
src������Ҷ�ͼ��
dst��������������Ķ�ֵͼ��
kernel����������
PCAfeature������ֵͼ������һ���������
*/
void PCAconv(Mat &src,Mat &dst,Mat &kernel,vector<float> &PCAfeature)
{
	if(src.depth() != CV_32F) 
		src.convertTo(src,CV_32F,1/255.);//תΪ0~1
	
	float temp = 0.0;

	filter2D(src,dst,src.depth(),kernel);

	for(int i=0;i<dst.rows;i++)
	{
		for(int j=0;j<dst.cols;j++)
		{
			//��ֵ����������0����Ϊ1
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


//��������������;���������feature map��Ȩ��ӣ��ϳ�һ��feature map��������һ����������
/*
src������Ҷ�ͼ��
dst��������Ӻ����յ�feature map
PCAfeature�������������
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