#include "opencvtest.h"

using namespace std;  
using namespace cv; 

CvSVM svm;//�½�һ��SVM�������������Է���

	//PCAnetѵ�������ĵ�һ��ĵ�һ�������
	Mat kernel1 = (Mat_<float>(5,5)<< -0.2498,   -0.2745,   -0.2828,   -0.2745,   -0.2498,
   -0.1580,   -0.1748,   -0.1804,   -0.1748,   -0.1580,
   -0.0004,   -0.0006,   -0.0006,   -0.0006,   -0.0004,
    0.1577,   0.1744,    0.1799,    0.1744,    0.1577,
    0.2509,    0.2754,    0.2835,    0.2754,    0.2509);

	//PCAnetѵ�������ĵ�һ��ĵڶ��������
	Mat kernel2 = (Mat_<float>(5,5)<<-0.2534,   -0.1515,   -0.0000,    0.1515,    0.2534,
   -0.2787,   -0.1686,   -0.0000,    0.1686,    0.2787,
   -0.2872,  -0.1745,   -0.0000,    0.1745,    0.2872,
   -0.2789,   -0.1693,    0.0000,    0.1693,    0.2789,
   -0.2531,   -0.1524,    0.0000,    0.1524,    0.2531);

	//PCAnetѵ�������ĵ�һ��ĵ����������
	Mat kernel3 = (Mat_<float>(5,5)<< 0.2537,    0.2126,    0.1865,    0.2126,    0.2537,
   -0.0237,   -0.1204,   -0.1674,   -0.1204,   -0.0237,
   -0.1742,   -0.3031,   -0.3617,   -0.3031,   -0.1742,
   -0.0254,   -0.1221,   -0.1693,   -0.1221,   -0.0254,
    0.2532,    0.2124,    0.1859,    0.2124,    0.2532);



//����ͳ��
/*
featureVec0��featureVec1�ֱ�Ϊ����ģ������������ͼ�⵽����������������
ChisquareVec�����������������֮��Ŀ���ͳ�ƽ��
*/
float Chisquare(vector<float> featureVec0,vector<float> featureVec1,vector<float> &ChisquareVec)
{
	int n = featureVec0.size();
	float sum = 0;
	float temp = 0;
	float weight = 1;//���ÿ���ͳ�Ƶ�Ȩ��
	int irow = 0;
	int icol = 0;

	for(int j=0;j<n;j++)
	{
		//���10*10�Ŀ�
		temp = featureVec0[j]-featureVec1[j];
		sum += temp*temp/(featureVec0[j]+featureVec1[j]+1);
		ChisquareVec.push_back(temp*temp/(featureVec0[j]+featureVec1[j]+1));
	}

	return sum;
}


//���Ŀ���ģ�����ƥ��
/*
pSrcImage����֡ͼ��
facewh�������淶�ߴ�
facebox����⵽���������ο���Ϣ
featureVec0��ģ���ֱ��ͼ��������
tempchi��SVM����ƥ�������ֵ
objbox���������ص���ƥ���ϵ�����Ŀ����ο���Ϣ
*/
vector<cv::Rect> boxmatch(IplImage *pSrcImage,Size facewh,vector<Rect> facebox,vector<float> featureVec0,float &tempchi)
{
	svm.load("model\\faceSVMclass3.xml", 0);//����ƥ��model

	int n = facebox.size();
	float tempro = 0;
	//float tempchi = -1.1;//����ͳ��

	int finalID = 999;
	Mat dst;
	Mat dst2;

	int label = 0;//Ĭ�ϲ���ͳһ����
	float decisionval = 0.0;//SVM�ж���������Ŷ�

	vector<cv::Rect> objbox;
	for(int i = 0; i<n; i++)
	{
		float ro = 0;//���ϵ��

		float chi = 0;//����ͳ��


		cvSetImageROI(pSrcImage,facebox[i]);
		Mat subimg(pSrcImage,0);
		resize(subimg, subimg, facewh);
		cvResetImageROI(pSrcImage);//���������������ROI 


		vector<float> featureVec1; 
		vector<float> histfeatureVec1; 
		if(subimg.channels() > 1)
			cvtColor(subimg,subimg,CV_BGR2GRAY);

		equalizeHist(subimg, subimg);//ֱ��ͼ���⻯
		#ifdef DEBUG
		imshow("subimg",subimg);
		#endif
		PCA2fea(subimg,dst,featureVec1);
		gray01(dst,histfeatureVec1);//����ֱ��ͼͳ��



		vector<float> ChisquareVec;
		//�ÿ���ͳ�������б�
		chi = Chisquare(featureVec0,histfeatureVec1,ChisquareVec);
		int featureVecSize = ChisquareVec.size();
		CvMat *testsampleMat = cvCreateMat(1, featureVecSize, CV_32FC1);
		for (int j=0; j<featureVecSize; j++)  
		{  		
			CV_MAT_ELEM( *testsampleMat, float, 0, j ) = ChisquareVec[j];
		}
		label = svm.predict(testsampleMat);
		decisionval = svm.predict(testsampleMat,true);


		if(label==1)
		{
			if(decisionval<tempchi)
			{
				finalID = i;
				tempchi = decisionval;
			}	
		}

		#ifdef DEBUG
		cout<<"label:"<<label<<endl;//ʶ����
		cout<<"decisionval:"<<decisionval<<endl;//���Ŷ�
		#endif

	}
	if(finalID<999)
	{
		objbox.push_back(facebox[finalID]);
	}

	return objbox;
}