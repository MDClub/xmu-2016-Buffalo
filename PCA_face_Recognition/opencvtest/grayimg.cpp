#include "opencvtest.h"

using namespace std;  
using namespace cv; 

//�ֿ�������map��ֱ��ͼ
/*
gray��������PCA2fea�����õ��ĵ��Ӻ����յ�feature map
featureVec���������map�ֿ�ͳ�Ƶ�ֱ��ͼ��������
*/
void gray01(Mat &gray,vector<float> &featureVec)
{
	int channels[] = {0};//��һ��ͨ��
	int histSize[] = {8};//ֱ��ͼ��Ϊ����
	float granges [] = {0.0,7.0};//�Ҷ�ͼȡֵ��Χ
	const float* ranges[] = {granges};

	int blocksize = 10;//�ֿ��С

	double minval=0;
	double maxval=0;

	float temp=0.0;

	for(int i=0;i<gray.rows;)
	{
		for(int j=0;j<gray.cols;)
		{
			//��ÿ�����ڵ�ÿһ����ֱ��ͼ������һ��������
			Rect roi(j, i, 1, blocksize);
			Mat roi_img = gray(roi);
			Mat hist;//ֱ��ͼ

			calcHist(&roi_img,1,channels,Mat(),hist,1,histSize,ranges,true,false);

			//minMaxLoc(hist, 0, &maxval, 0, 0);//�����ֵ

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