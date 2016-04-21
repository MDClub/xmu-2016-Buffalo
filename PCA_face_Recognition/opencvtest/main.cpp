#include "opencvtest.h"

using namespace std;
using namespace cv;


const char *pcascadeName = "model\\haarcascade_frontalface_alt_tree.xml"; //������������������
CvCapture *capture;
int FrameNum = 0;//��ʼ֡��

void ON_Change(int n)    
{    
    cvSetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES,n);     //������Ƶ�ߵ�posλ��    
} 

//������
int main(int argc, const char** argv)  
{  
	if(argc<4)
	{
		cout<<"����������ٻ��ʽ����,��˶ԣ�"<<endl;
		system("pause");
	}

	capture = cvCreateFileCapture(argv[1]);//��ȡ��Ƶ
	Mat oimg = imread(argv[2]);//��ȡĿ��ͼƬ
	const char* objname = argv[3];//Ŀ����������
	int facesize = 64;//�������Ҫ������С����
	float faceweight = -1.5;//SVM�ķ�����ֵ


	facesize = atoi(argv[4]);
	faceweight = atof(argv[5]);//SVM�ķ�����ֵ

	cout<<"������Ƶ:"<<argv[1]<<endl;
	cout<<"����ͼ��:"<<argv[2]<<endl;
	cout<<"Ŀ������:"<<argv[3]<<"(��֧����ʾӢ����ĸ������)"<<endl;
	cout<<"�������Ҫ������С�����ߴ�:"<<argv[4]<<"(Ĭ��Ϊ64)"<<endl;
	cout<<"SVM�ķ�����ֵ:"<<argv[5]<<"(Ĭ��Ϊ-2.5)"<<endl;

	CvSize minsize = cvSize(facesize,facesize);

	IplImage *curframe = NULL;//��Ƶ��ǰ֡ͼ��
	uchar key = false;//����������ͣ
	bool pause = false;//�Ƿ���ͣ
	vector<Rect> facebox;//�������������ο���Ϣ
	vector<Rect> objbox;//ƥ���õ���Ŀ���
	Size facewh = Size(80,80);//��һ�������ߴ��С

	int TimeStart, TimeEnd;//����ÿ֡��ʱ

	// load the Haar classifier  
    CvHaarClassifierCascade *pHaarClassCascade;  
    pHaarClassCascade = (CvHaarClassifierCascade*)cvLoad(pcascadeName); 

	IplImage timg = IplImage(oimg);
	facebox = DetectAndMark(&timg,pHaarClassCascade,1,minsize,true);//��������ģ��
	IplImage* Grayimg = cvCreateImage(cvGetSize(&timg),IPL_DEPTH_8U,1);
	if(timg.nChannels > 1)
		cvCvtColor(&timg,Grayimg,CV_BGR2GRAY);//תΪ�Ҷ�ͼ

	cvSetImageROI(Grayimg,facebox[0]);
	cvSaveImage("obj.jpg",Grayimg);//����Ҷ�ͼ����
	Mat img = Grayimg;
	cvResetImageROI(Grayimg);//���������������ROI

	resize(img, img, facewh);//�淶��������С


	vector<float> featureVec0;
	vector<float> histfeatureVec0;
	Mat dst;

	//equalizeHist(img, img);//ֱ��ͼ���⻯

	const char *pstrWindowsTitle = "objface";  
    cvNamedWindow(pstrWindowsTitle, CV_WINDOW_AUTOSIZE);  
	imshow(pstrWindowsTitle, img);//��ʾ����ģ��

	PCA2fea(img,dst,featureVec0);

	gray01(dst,histfeatureVec0);//����ֱ��ͼͳ��

	int frames = (int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);//��ȡ��Ƶ֡��
	printf("��Ƶ��֡��: %d\n", frames);
	cout<<"����P���Կ�����Ƶ����ͣ/����"<<endl;

	cvNamedWindow("FaceDetect Demo",CV_WINDOW_NORMAL);
	curframe = cvQueryFrame(capture); //ץȡһ֡
	cvShowImage("FaceDetect Demo", curframe);
	cvCreateTrackbar("frame","FaceDetect Demo",&FrameNum,frames,ON_Change);//�����������ؼ�  

	FILE* fp = fopen("objectDetectframe.txt","wb");//�洢��Ƶ�г���Ŀ���֡��

	while (capture)
	{
		TimeStart = GetTickCount();  //������ʱ��
		curframe = cvQueryFrame(capture); //ץȡһ֡

		facebox = DetectAndMark(curframe,pHaarClassCascade,2,minsize,false);//��ÿ֡ͼ�����������⣬�����ؾ��ο���Ϣ

		if(facebox.size()>0)
		{
			//�Լ�⵽Ŀ���������ƥ��
			float tempchi = faceweight;
			objbox = boxmatch(curframe,facewh,facebox,histfeatureVec0,tempchi);

			char text[40] = "";
			CvFont font;

			if(objbox.size()>0){

			for(int idx = 0;idx<objbox.size();idx++)
			{
				cvRectangle(curframe, cvPoint(facebox[idx].x,facebox[idx].y),
				cvPoint(facebox[idx].x+facebox[idx].width,facebox[idx].y+facebox[idx].height),CV_RGB(255,20,0), 2);//��ͼƬ�ϻ��ƾ��ο�
				sprintf(text,"%s:%.3f",objname,-tempchi);//��ʾĿ������ּ����Ŷ�
				cvInitFont(&font,CV_FONT_HERSHEY_COMPLEX, 0.8, 0.8, 0, 1, 8); 
				cvPutText(curframe,text, cvPoint(facebox[idx].x,facebox[idx].y), &font, CV_RGB(0, 255, 255));//��עĿ��
				cvShowImage("FaceDetect Demo", curframe);
			}

			fprintf(fp,"%d \n",FrameNum);

			}

		}
		FrameNum++;
		
		cvShowImage("FaceDetect Demo", curframe);
		//cvSetTrackbarPos("frame","FaceDetect Demo",FrameNum);//���ý�����λ��,����ܺ�ʱ
		
		TimeEnd = GetTickCount();

		#ifdef DEBUG
		printf("ÿ֡��ʱ: %d ms\n", TimeEnd - TimeStart); //ÿ֡�ܴ���ʱ��
		#endif

		if(FrameNum == frames-1)//��ֹ֡��
			break;

		//����P�л���ͣ�Ͳ���
		key = cvWaitKey(1);
		if(key == 'p') pause = true;
		while(pause)
			if(cvWaitKey(0)=='p')
				pause = false;		
	}

	fclose(fp);

	cvWaitKey(0);
	cvReleaseCapture(&capture); //�ͷ���Ƶ�ռ�    
    cvDestroyWindow("FaceDetect Demo");    //���ٴ��� 
	img.release();//�ͷ�ͼ��
	cvDestroyWindow("objface");

    return 0;
}



