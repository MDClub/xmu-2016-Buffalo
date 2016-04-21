#include "opencvtest.h"

using namespace std;  
using namespace cv; 

extern Mat kernel1;
extern Mat kernel2;
extern Mat kernel3;
extern Mat kernel4;


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
	//filter2D(src,dst,src.depth(),kernel);
	Mat blockimg = Mat_<float>(25,76*76);
	float blocksum;
	int blrows;
	int blcols = 0;
	//Ū��25*��76*76���ľ���
	for(int i=0;i<src.rows-4;i++)
	{
		for(int j=0;j<src.cols-4;j++)
		{
			blocksum = 0.0;

			for(int k1=0;k1<5;k1++)
			{
				for(int k2=0;k2<5;k2++)
				{
					blocksum += src.at<float>(i+k1,j+k2);
				}
			}
			float blockmean = blocksum/25;
			blrows = 0;
			for(int k1=0;k1<5;k1++)
			{
				for(int k2=0;k2<5;k2++)
				{
					blockimg.at<float>(blrows,blcols) = src.at<float>(i+k1,j+k2)-blockmean;
					blrows++;
				}
			}
			blcols++;
		}
	}

	Mat feat = kernel*blockimg;
	
	dst = feat.reshape(0,76);
	int border = 2;
	copyMakeBorder(dst, dst, border, border,
               border, border, BORDER_REPLICATE);

	#pragma omp parallel for
	for(int i=0;i<dst.rows;i++)
	{
		for(int j=0;j<dst.cols;j++)
		{
			//��ֵ��
			temp = dst.at<float>(i,j);
			if(temp>0.00)
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
	vector<float> PCAfeature4;

	Mat dst1;
	Mat dst2;
	Mat dst3;
	Mat dst4;


	PCAconv(src,dst1,kernel1,PCAfeature1);
	PCAconv(src,dst2,kernel2,PCAfeature2);
	PCAconv(src,dst3,kernel3,PCAfeature3);
	PCAconv(src,dst4,kernel4,PCAfeature4);

	addWeighted(dst1, 1.0, dst2, 2.0, 0.0, dst);
	addWeighted(dst, 1.0, dst3, 4.0, 0.0, dst);
	addWeighted(dst, 1.0, dst4, 8.0, 0.0, dst);


	for(int i=0;i<dst.rows;i++)
	{
		for(int j=0;j<dst.cols;j++)
		{
			PCAfeature.push_back(dst.at<float>(i,j));
		}
	}

}