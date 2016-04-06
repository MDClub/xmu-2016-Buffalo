#include "opencvtest.h"

using namespace std;  
using namespace cv; 



vector<cv::Rect> boxmatch(IplImage *pSrcImage,Size facewh,HOGDescriptor hog,vector<Rect> facebox,vector<float> featureVec0)
{
	int n = facebox.size();
	float tempro = 0;
	int finalID = 999;

	vector<cv::Rect> objbox;
	for(int i = 0; i<n; i++)
	{
		float cha0 = 0;
		float cha1 = 0;
		float ro = 0;//相关系数
		float fenzi = 0;
		float fenmu = 0.001;
		float cha2_0 = 0;
		float cha2_1 = 0;

		cvSetImageROI(pSrcImage,facebox[i]);
		Mat subimg(pSrcImage,0);
		resize(subimg, subimg, facewh);
		cvResetImageROI(pSrcImage);//复制完后重新设置ROI 


		vector<float> featureVec1; 
		gray01(subimg,featureVec1);



		//使用相关系数来做判别吧？
		float sum0 = accumulate(featureVec0.begin() , featureVec0.end() , 0.f);//求和
		float sum1 = accumulate(featureVec1.begin() , featureVec1.end() , 0.f);//求和
		int n = featureVec0.size();

		float mean0 = sum0/n;
		float mean1 = sum1/n;

		for(int j=0;j<n;j++)
		{
			cha0 = featureVec0[j]-mean0;		
			cha1 = featureVec1[j]-mean1;
			fenzi += cha0*cha1;
			cha2_0 += cha0*cha0;
			cha2_1 += cha1*cha1;
		}
		fenmu += sqrt(cha2_0*cha2_1);
		ro = fenzi/fenmu;

		if(ro>0.4 && facebox[i].height>40)
		{
			if(ro>tempro)
			{
				finalID = i;
				tempro = ro;
			}
			/*
			char text[20] = "";
			CvFont font;
			cvRectangle(pSrcImage, cvPoint(facebox[i].x,facebox[i].y),
				cvPoint(facebox[i].x+facebox[i].width,facebox[i].y+facebox[i].height),CV_RGB(255,20,0), 2);//在图片上绘制矩形框
			sprintf(text,"Object");
			cvInitFont(&font,CV_FONT_HERSHEY_COMPLEX, 0.5, 0.5, 0, 1, 8); 
			cvPutText(pSrcImage,text, cvPoint(facebox[i].x,facebox[i].y), &font, CV_RGB(0, 255, 255));
			const char *pstrWindowsTitle = "FaceDetect Demo";
			cvNamedWindow(pstrWindowsTitle, CV_WINDOW_AUTOSIZE);  
			cvShowImage(pstrWindowsTitle, pSrcImage);
			*/
		
		}	
		cout<<"cha:"<<ro<<endl;//差值小于某阈值的可以认为是同一个？？？

	}
	if(finalID<999)
	{
		objbox.push_back(facebox[finalID]);
	}

	return objbox;
}