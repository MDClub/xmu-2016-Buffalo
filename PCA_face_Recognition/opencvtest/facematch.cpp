#include "opencvtest.h"

using namespace std;  
using namespace cv; 

CvSVM svm;//新建一个SVM用作人脸相似性分类

	//PCAnet训练出来的第一层的第一个卷积核
	Mat kernel1 = (Mat_<float>(5,5)<< -0.2498,   -0.2745,   -0.2828,   -0.2745,   -0.2498,
   -0.1580,   -0.1748,   -0.1804,   -0.1748,   -0.1580,
   -0.0004,   -0.0006,   -0.0006,   -0.0006,   -0.0004,
    0.1577,   0.1744,    0.1799,    0.1744,    0.1577,
    0.2509,    0.2754,    0.2835,    0.2754,    0.2509);

	//PCAnet训练出来的第一层的第二个卷积核
	Mat kernel2 = (Mat_<float>(5,5)<<-0.2534,   -0.1515,   -0.0000,    0.1515,    0.2534,
   -0.2787,   -0.1686,   -0.0000,    0.1686,    0.2787,
   -0.2872,  -0.1745,   -0.0000,    0.1745,    0.2872,
   -0.2789,   -0.1693,    0.0000,    0.1693,    0.2789,
   -0.2531,   -0.1524,    0.0000,    0.1524,    0.2531);

	//PCAnet训练出来的第一层的第三个卷积核
	Mat kernel3 = (Mat_<float>(5,5)<< 0.2537,    0.2126,    0.1865,    0.2126,    0.2537,
   -0.0237,   -0.1204,   -0.1674,   -0.1204,   -0.0237,
   -0.1742,   -0.3031,   -0.3617,   -0.3031,   -0.1742,
   -0.0254,   -0.1221,   -0.1693,   -0.1221,   -0.0254,
    0.2532,    0.2124,    0.1859,    0.2124,    0.2532);



//卡方统计
/*
featureVec0和featureVec1分别为人脸模板的特征向量和检测到的人脸的特征向量
ChisquareVec：输出两个特征向量之间的卡方统计结果
*/
float Chisquare(vector<float> featureVec0,vector<float> featureVec1,vector<float> &ChisquareVec)
{
	int n = featureVec0.size();
	float sum = 0;
	float temp = 0;
	float weight = 1;//设置卡方统计的权重
	int irow = 0;
	int icol = 0;

	for(int j=0;j<n;j++)
	{
		//针对10*10的块
		temp = featureVec0[j]-featureVec1[j];
		sum += temp*temp/(featureVec0[j]+featureVec1[j]+1);
		ChisquareVec.push_back(temp*temp/(featureVec0[j]+featureVec1[j]+1));
	}

	return sum;
}


//检测目标跟模板进行匹配
/*
pSrcImage：单帧图像
facewh：人脸规范尺寸
facebox：检测到的人脸矩形框信息
featureVec0：模板的直方图特征向量
tempchi：SVM人脸匹配分类阈值
objbox：函数返回的是匹配上的人脸目标矩形框信息
*/
vector<cv::Rect> boxmatch(IplImage *pSrcImage,Size facewh,vector<Rect> facebox,vector<float> featureVec0,float &tempchi)
{
	svm.load("model\\faceSVMclass3.xml", 0);//人脸匹配model

	int n = facebox.size();
	float tempro = 0;
	//float tempchi = -1.1;//卡方统计

	int finalID = 999;
	Mat dst;
	Mat dst2;

	int label = 0;//默认不是统一个人
	float decisionval = 0.0;//SVM判定结果的置信度

	vector<cv::Rect> objbox;
	for(int i = 0; i<n; i++)
	{
		float ro = 0;//相关系数

		float chi = 0;//卡方统计


		cvSetImageROI(pSrcImage,facebox[i]);
		Mat subimg(pSrcImage,0);
		resize(subimg, subimg, facewh);
		cvResetImageROI(pSrcImage);//复制完后重新设置ROI 


		vector<float> featureVec1; 
		vector<float> histfeatureVec1; 
		if(subimg.channels() > 1)
			cvtColor(subimg,subimg,CV_BGR2GRAY);

		equalizeHist(subimg, subimg);//直方图均衡化
		#ifdef DEBUG
		imshow("subimg",subimg);
		#endif
		PCA2fea(subimg,dst,featureVec1);
		gray01(dst,histfeatureVec1);//特征直方图统计



		vector<float> ChisquareVec;
		//用卡方统计来做判别？
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
		cout<<"label:"<<label<<endl;//识别结果
		cout<<"decisionval:"<<decisionval<<endl;//置信度
		#endif

	}
	if(finalID<999)
	{
		objbox.push_back(facebox[finalID]);
	}

	return objbox;
}