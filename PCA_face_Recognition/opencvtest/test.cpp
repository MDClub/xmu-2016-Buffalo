#include "opencvtest.h"

using namespace std;
using namespace cv;

const char *pcascadeName = "haarcascade_frontalface_alt.xml"; //������������������
CvCapture *capture;

int FrameNum = 0;//֡��
void ON_Change(int n)    
{    
    cvSetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES,n);     //������Ƶ�ߵ�posλ��    
} 

int main(int argc, const char** argv)  
{  
	IplImage *curframe = NULL;//��Ƶ��ǰ֡ͼ��
	uchar key = false;//����������ͣ
	bool pause = false;//�Ƿ���ͣ
	vector<Rect> facebox;//�������������ο���Ϣ
	vector<cv::Rect> objbox;//Ŀ���
	IplImage* subimg = NULL;//

	Size facewh = Size(96,96);


	HOGDescriptor hog(facewh, cv::Size(16,16), cv::Size(4,4), cv::Size(8,8), 9,-1,-1,HOGDescriptor::L2Hys,0.2,true);//��������HOG����

	// load the Haar classifier  
    CvHaarClassifierCascade *pHaarClassCascade;  
    pHaarClassCascade = (CvHaarClassifierCascade*)cvLoad(pcascadeName); 

	Mat oimg = imread("objimg1.jpg");//��ȡĿ��ͼƬ
	//Mat oimg = imread(argv[2]);//��ȡĿ��ͼƬ
	IplImage timg = IplImage(oimg);
	facebox = DetectAndMark(&timg,pHaarClassCascade,1);
	IplImage* Grayimg = cvCreateImage(cvGetSize(&timg),IPL_DEPTH_8U,1);
	cvCvtColor(&timg,Grayimg,CV_BGR2GRAY);
	cvSetImageROI(Grayimg,facebox[0]);
	cvSaveImage("obj.jpg",Grayimg);//����Ҷ�ͼ����
	Mat img = Grayimg;
	cvResetImageROI(Grayimg);//���������������ROI
	/*
	cvSetImageROI(&timg,facebox[0]);
	cvSaveImage("obj.jpg",&timg);
	cvResetImageROI(&timg);//���������������ROI
	*/
	
	//Mat img = imread("obj.jpg");
	resize(img, img, facewh);
	Mat Mirrorimg;
	hMirrorTrans(img, Mirrorimg);
	imwrite("obj_.jpg",Mirrorimg);

	const char *pstrWindowsTitle = "objface";  
    cvNamedWindow(pstrWindowsTitle, CV_WINDOW_AUTOSIZE);  
    imshow(pstrWindowsTitle, img);

	vector<float> featureVec0; 
	hog.compute(img,featureVec0,cv::Size(8,8));//����Ŀ��1��HOG����
	vector<float> featureVec0_; 
	hog.compute(Mirrorimg,featureVec0_,cv::Size(8,8));//����Ŀ��2��HOG����


	//capture = cvCreateFileCapture(argv[1]);
	capture = cvCreateFileCapture("threep.mp4");

	int frames = (int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);//��ȡ��Ƶ֡��
	printf("total frames is %d\n", frames);
	cout<<"����P���Կ�����Ƶ����ͣ/����"<<endl;

	cvNamedWindow("FaceDetect Demo",CV_WINDOW_AUTOSIZE);
	curframe = cvQueryFrame(capture); //ץȡһ֡
	cvShowImage("FaceDetect Demo", curframe);

	cvCreateTrackbar("frame","FaceDetect Demo",&FrameNum,frames,ON_Change);//����������  


	while (capture)
	{
		curframe = cvQueryFrame(capture); //ץȡһ֡

		facebox = DetectAndMark(curframe,pHaarClassCascade,2);//����������⣬�����ؾ��ο���Ϣ
		if(facebox.size()>0)
		{
			printf("face number: %d\n",facebox.size());
			//ƥ��
			objbox = boxmatch(curframe,facewh,hog,facebox,featureVec0,featureVec0_);

			char text[20] = "";
			CvFont font;

			for(int idx = 0;idx<objbox.size();idx++)
			{
				cvRectangle(curframe, cvPoint(facebox[idx].x,facebox[idx].y),
				cvPoint(facebox[idx].x+facebox[idx].width,facebox[idx].y+facebox[idx].height),CV_RGB(255,20,0), 2);//��ͼƬ�ϻ��ƾ��ο�
				sprintf(text,"Object");
				cvInitFont(&font,CV_FONT_HERSHEY_COMPLEX, 0.5, 0.5, 0, 1, 8); 
				cvPutText(curframe,text, cvPoint(facebox[idx].x,facebox[idx].y), &font, CV_RGB(0, 255, 255));
				cvShowImage("FaceDetect Demo", curframe);
			}

		}
		FrameNum++;
		
		cvShowImage("FaceDetect Demo", curframe);
		cvSetTrackbarPos("frame","FaceDetect Demo",FrameNum);//���ý�����λ��
		

		if(FrameNum == frames-1)//��ֹ֡��
			break;

		//����P�л���ͣ�Ͳ���
		key = cvWaitKey(1);
		if(key == 'p') pause = true;
		while(pause)
			if(cvWaitKey(0)=='p')
				pause = false;		
	}


	/*��ͼƬ�����������
	const char *pImageName = argv[1];//�����ͼƬ	
	IplImage *pSrcImage = cvLoadImage(pImageName, CV_LOAD_IMAGE_UNCHANGED);
	DetectAndMark(pSrcImage,pHaarClassCascade);
	*/
	cvWaitKey(0);
	cvReleaseCapture(&capture); //�ͷ���Ƶ�ռ�    
    cvDestroyWindow("FaceDetect Demo");    //���ٴ���

    return 0;
}



