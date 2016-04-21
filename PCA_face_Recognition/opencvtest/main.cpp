#include "opencvtest.h"

using namespace std;
using namespace cv;


const char *pcascadeName = "model\\haarcascade_frontalface_alt_tree.xml"; //正面人脸级联分类器
CvCapture *capture;
int FrameNum = 0;//起始帧号

void ON_Change(int n)    
{    
    cvSetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES,n);     //设置视频走到pos位置    
} 

//主程序
int main(int argc, const char** argv)  
{  
	if(argc<4)
	{
		cout<<"输入参数过少或格式不对,请核对！"<<endl;
		system("pause");
	}

	capture = cvCreateFileCapture(argv[1]);//读取视频
	Mat oimg = imread(argv[2]);//读取目标图片
	const char* objname = argv[3];//目标人物姓名
	int facesize = 64;//检测器需要检测的最小人脸
	float faceweight = -1.5;//SVM的分类阈值


	facesize = atoi(argv[4]);
	faceweight = atof(argv[5]);//SVM的分类阈值

	cout<<"输入视频:"<<argv[1]<<endl;
	cout<<"输入图像:"<<argv[2]<<endl;
	cout<<"目标名字:"<<argv[3]<<"(仅支持显示英文字母或数字)"<<endl;
	cout<<"检测器需要检测的最小人脸尺寸:"<<argv[4]<<"(默认为64)"<<endl;
	cout<<"SVM的分类阈值:"<<argv[5]<<"(默认为-2.5)"<<endl;

	CvSize minsize = cvSize(facesize,facesize);

	IplImage *curframe = NULL;//视频当前帧图像
	uchar key = false;//用来设置暂停
	bool pause = false;//是否暂停
	vector<Rect> facebox;//检测出的人脸矩形框信息
	vector<Rect> objbox;//匹配后得到的目标框
	Size facewh = Size(80,80);//归一化人脸尺寸大小

	int TimeStart, TimeEnd;//计算每帧耗时

	// load the Haar classifier  
    CvHaarClassifierCascade *pHaarClassCascade;  
    pHaarClassCascade = (CvHaarClassifierCascade*)cvLoad(pcascadeName); 

	IplImage timg = IplImage(oimg);
	facebox = DetectAndMark(&timg,pHaarClassCascade,1,minsize,true);//检测出人脸模板
	IplImage* Grayimg = cvCreateImage(cvGetSize(&timg),IPL_DEPTH_8U,1);
	if(timg.nChannels > 1)
		cvCvtColor(&timg,Grayimg,CV_BGR2GRAY);//转为灰度图

	cvSetImageROI(Grayimg,facebox[0]);
	cvSaveImage("obj.jpg",Grayimg);//保存灰度图人脸
	Mat img = Grayimg;
	cvResetImageROI(Grayimg);//复制完后重新设置ROI

	resize(img, img, facewh);//规范化人脸大小


	vector<float> featureVec0;
	vector<float> histfeatureVec0;
	Mat dst;

	//equalizeHist(img, img);//直方图均衡化

	const char *pstrWindowsTitle = "objface";  
    cvNamedWindow(pstrWindowsTitle, CV_WINDOW_AUTOSIZE);  
	imshow(pstrWindowsTitle, img);//显示人脸模板

	PCA2fea(img,dst,featureVec0);

	gray01(dst,histfeatureVec0);//特征直方图统计

	int frames = (int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);//获取视频帧数
	printf("视频总帧数: %d\n", frames);
	cout<<"按键P可以控制视频的暂停/播放"<<endl;

	cvNamedWindow("FaceDetect Demo",CV_WINDOW_NORMAL);
	curframe = cvQueryFrame(capture); //抓取一帧
	cvShowImage("FaceDetect Demo", curframe);
	cvCreateTrackbar("frame","FaceDetect Demo",&FrameNum,frames,ON_Change);//创建滑动条控件  

	FILE* fp = fopen("objectDetectframe.txt","wb");//存储视频中出现目标的帧号

	while (capture)
	{
		TimeStart = GetTickCount();  //计算检测时间
		curframe = cvQueryFrame(capture); //抓取一帧

		facebox = DetectAndMark(curframe,pHaarClassCascade,2,minsize,false);//对每帧图像进行人脸检测，并返回矩形框信息

		if(facebox.size()>0)
		{
			//对检测到目标进行特征匹配
			float tempchi = faceweight;
			objbox = boxmatch(curframe,facewh,facebox,histfeatureVec0,tempchi);

			char text[40] = "";
			CvFont font;

			if(objbox.size()>0){

			for(int idx = 0;idx<objbox.size();idx++)
			{
				cvRectangle(curframe, cvPoint(facebox[idx].x,facebox[idx].y),
				cvPoint(facebox[idx].x+facebox[idx].width,facebox[idx].y+facebox[idx].height),CV_RGB(255,20,0), 2);//在图片上绘制矩形框
				sprintf(text,"%s:%.3f",objname,-tempchi);//显示目标的名字及置信度
				cvInitFont(&font,CV_FONT_HERSHEY_COMPLEX, 0.8, 0.8, 0, 1, 8); 
				cvPutText(curframe,text, cvPoint(facebox[idx].x,facebox[idx].y), &font, CV_RGB(0, 255, 255));//标注目标
				cvShowImage("FaceDetect Demo", curframe);
			}

			fprintf(fp,"%d \n",FrameNum);

			}

		}
		FrameNum++;
		
		cvShowImage("FaceDetect Demo", curframe);
		//cvSetTrackbarPos("frame","FaceDetect Demo",FrameNum);//设置进度条位置,这个很耗时
		
		TimeEnd = GetTickCount();

		#ifdef DEBUG
		printf("每帧耗时: %d ms\n", TimeEnd - TimeStart); //每帧总处理时间
		#endif

		if(FrameNum == frames-1)//截止帧数
			break;

		//按键P切换暂停和播放
		key = cvWaitKey(1);
		if(key == 'p') pause = true;
		while(pause)
			if(cvWaitKey(0)=='p')
				pause = false;		
	}

	fclose(fp);

	cvWaitKey(0);
	cvReleaseCapture(&capture); //释放视频空间    
    cvDestroyWindow("FaceDetect Demo");    //销毁窗口 
	img.release();//释放图像
	cvDestroyWindow("objface");

    return 0;
}



