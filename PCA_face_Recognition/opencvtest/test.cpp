#include "opencvtest.h"

using namespace std;
using namespace cv;

const char *pcascadeName = "haarcascade_frontalface_alt.xml"; //正面人脸级联分类器
CvCapture *capture;

int FrameNum = 0;//帧号
void ON_Change(int n)    
{    
    cvSetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES,n);     //设置视频走到pos位置    
} 

int main(int argc, const char** argv)  
{  
	IplImage *curframe = NULL;//视频当前帧图像
	uchar key = false;//用来设置暂停
	bool pause = false;//是否暂停
	vector<Rect> facebox;//检测出的人脸矩形框信息
	vector<cv::Rect> objbox;//目标框
	IplImage* subimg = NULL;//

	Size facewh = Size(96,96);


	HOGDescriptor hog(facewh, cv::Size(16,16), cv::Size(4,4), cv::Size(8,8), 9,-1,-1,HOGDescriptor::L2Hys,0.2,true);//计算人脸HOG特征

	// load the Haar classifier  
    CvHaarClassifierCascade *pHaarClassCascade;  
    pHaarClassCascade = (CvHaarClassifierCascade*)cvLoad(pcascadeName); 

	Mat oimg = imread("objimg1.jpg");//读取目标图片
	//Mat oimg = imread(argv[2]);//读取目标图片
	IplImage timg = IplImage(oimg);
	facebox = DetectAndMark(&timg,pHaarClassCascade,1);
	IplImage* Grayimg = cvCreateImage(cvGetSize(&timg),IPL_DEPTH_8U,1);
	cvCvtColor(&timg,Grayimg,CV_BGR2GRAY);
	cvSetImageROI(Grayimg,facebox[0]);
	cvSaveImage("obj.jpg",Grayimg);//保存灰度图人脸
	Mat img = Grayimg;
	cvResetImageROI(Grayimg);//复制完后重新设置ROI
	/*
	cvSetImageROI(&timg,facebox[0]);
	cvSaveImage("obj.jpg",&timg);
	cvResetImageROI(&timg);//复制完后重新设置ROI
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
	hog.compute(img,featureVec0,cv::Size(8,8));//计算目标1的HOG特征
	vector<float> featureVec0_; 
	hog.compute(Mirrorimg,featureVec0_,cv::Size(8,8));//计算目标2的HOG特征


	//capture = cvCreateFileCapture(argv[1]);
	capture = cvCreateFileCapture("threep.mp4");

	int frames = (int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);//获取视频帧数
	printf("total frames is %d\n", frames);
	cout<<"按键P可以控制视频的暂停/播放"<<endl;

	cvNamedWindow("FaceDetect Demo",CV_WINDOW_AUTOSIZE);
	curframe = cvQueryFrame(capture); //抓取一帧
	cvShowImage("FaceDetect Demo", curframe);

	cvCreateTrackbar("frame","FaceDetect Demo",&FrameNum,frames,ON_Change);//创建滚动条  


	while (capture)
	{
		curframe = cvQueryFrame(capture); //抓取一帧

		facebox = DetectAndMark(curframe,pHaarClassCascade,2);//进行人脸检测，并返回矩形框信息
		if(facebox.size()>0)
		{
			printf("face number: %d\n",facebox.size());
			//匹配
			objbox = boxmatch(curframe,facewh,hog,facebox,featureVec0,featureVec0_);

			char text[20] = "";
			CvFont font;

			for(int idx = 0;idx<objbox.size();idx++)
			{
				cvRectangle(curframe, cvPoint(facebox[idx].x,facebox[idx].y),
				cvPoint(facebox[idx].x+facebox[idx].width,facebox[idx].y+facebox[idx].height),CV_RGB(255,20,0), 2);//在图片上绘制矩形框
				sprintf(text,"Object");
				cvInitFont(&font,CV_FONT_HERSHEY_COMPLEX, 0.5, 0.5, 0, 1, 8); 
				cvPutText(curframe,text, cvPoint(facebox[idx].x,facebox[idx].y), &font, CV_RGB(0, 255, 255));
				cvShowImage("FaceDetect Demo", curframe);
			}

		}
		FrameNum++;
		
		cvShowImage("FaceDetect Demo", curframe);
		cvSetTrackbarPos("frame","FaceDetect Demo",FrameNum);//设置进度条位置
		

		if(FrameNum == frames-1)//截止帧数
			break;

		//按键P切换暂停和播放
		key = cvWaitKey(1);
		if(key == 'p') pause = true;
		while(pause)
			if(cvWaitKey(0)=='p')
				pause = false;		
	}


	/*对图片进行人脸检测
	const char *pImageName = argv[1];//待检测图片	
	IplImage *pSrcImage = cvLoadImage(pImageName, CV_LOAD_IMAGE_UNCHANGED);
	DetectAndMark(pSrcImage,pHaarClassCascade);
	*/
	cvWaitKey(0);
	cvReleaseCapture(&capture); //释放视频空间    
    cvDestroyWindow("FaceDetect Demo");    //销毁窗口

    return 0;
}



