#include "opencvtest.h"

using namespace std;  
using namespace cv; 


//检测单帧图像里的人脸区域
/*
参数说明
pSrcImage:单帧图像
pHaarClassCascade：OpenCV自带的类Haar人脸检测器
min_neighbor：判定是否为人脸的标准，取值1、2...，越大，表明该区域是人脸的可能性越大
minsize：需要检测的最小人脸尺寸
templet：true表示是对模板图片进行检测，flase表示是对视频图片进行检测
*/
vector<Rect> DetectAndMark(IplImage *pSrcImage,CvHaarClassifierCascade *pHaarClassCascade,int min_neighbor,CvSize minsize,bool templet)
{
	vector<Rect> facebox;

    //load the test image  
    IplImage *pGrayImage = cvCreateImage(cvGetSize(pSrcImage), IPL_DEPTH_8U, 1);  
	if(pSrcImage == NULL || pGrayImage == NULL)
	{
		 printf("can't load image!\n");
		 return facebox;
	}
    cvCvtColor(pSrcImage, pGrayImage, CV_BGR2GRAY); //转为灰度图 
  
    if (pHaarClassCascade != NULL && pSrcImage != NULL && pGrayImage != NULL)  
    {          
        CvMemStorage *pcvMemStorage = cvCreateMemStorage(0);  
        cvClearMemStorage(pcvMemStorage);  

        //detect the face
        int TimeStart, TimeEnd;  
        TimeStart = GetTickCount();  //计算检测时间
        CvSeq *pcvSeqFaces = cvHaarDetectObjects(pGrayImage, pHaarClassCascade, pcvMemStorage, 1.1, min_neighbor, CV_HAAR_DO_CANNY_PRUNING,minsize);  
        TimeEnd = GetTickCount();  
  
		#ifdef DEBUG
        printf("Spending Time: %d ms\n", TimeEnd - TimeStart);//每帧图像的人脸检测时间
        #endif

        //mark the face   
        for(int i = 0; i <pcvSeqFaces->total; i++)  
        {  
            CvRect* r = (CvRect*)cvGetSeqElem(pcvSeqFaces, i);  

			//检测到的矩形框上、左、右边框向中心缩进5%
            Rect tempbox;
			tempbox.x = r->x + floor((r->width)*0.05);
			tempbox.y = r->y + floor((r->width)*0.05);
			tempbox.width = r->width - 2*floor((r->width)*0.05);
			tempbox.height = r->height - 2*floor((r->width)*0.05);
			facebox.push_back(tempbox);

			//对检测到的图片进行多方位位置调整
			if(!templet)
			{
			tempbox.x = r->x + floor((r->width)*0.1);
			tempbox.y = r->y + floor((r->width)*0.1);
			tempbox.width = r->width - 2*floor((r->width)*0.1);
			tempbox.height = r->height - 2*floor((r->width)*0.1);
			facebox.push_back(tempbox);

			tempbox.x = r->x;
			tempbox.y = r->y;
			tempbox.width = r->width;
			tempbox.height = r->height;
			facebox.push_back(tempbox);

			tempbox.x = r->x - floor((r->width)*0.03);
			tempbox.y = r->y - floor((r->width)*0.03);
			tempbox.width = r->width + 2*floor((r->width)*0.03);
			tempbox.height = r->height + 2*floor((r->width)*0.03);
			facebox.push_back(tempbox);

			tempbox.x = r->x - floor((r->width)*0.08);
			tempbox.y = r->y;
			tempbox.width = r->width;
			tempbox.height = r->height;
			facebox.push_back(tempbox);

			tempbox.x = r->x + floor((r->width)*0.08);
			tempbox.y = r->y;
			tempbox.width = r->width;
			tempbox.height = r->height;
			facebox.push_back(tempbox);

			tempbox.x = r->x;
			tempbox.y = r->y + floor((r->width)*0.08);
			tempbox.width = r->width;
			tempbox.height = r->height;
			facebox.push_back(tempbox);

			tempbox.x = r->x;
			tempbox.y = r->y - floor((r->width)*0.08);
			tempbox.width = r->width;
			tempbox.height = r->height;
			facebox.push_back(tempbox);

			tempbox.x = r->x - floor((r->width)*0.12);
			tempbox.y = r->y;
			tempbox.width = r->width;
			tempbox.height = r->height;
			facebox.push_back(tempbox);

			tempbox.x = r->x + floor((r->width)*0.12);
			tempbox.y = r->y;
			tempbox.width = r->width;
			tempbox.height = r->height;
			facebox.push_back(tempbox);

			tempbox.x = r->x;
			tempbox.y = r->y + floor((r->width)*0.12);
			tempbox.width = r->width;
			tempbox.height = r->height;
			facebox.push_back(tempbox);

			tempbox.x = r->x;
			tempbox.y = r->y - floor((r->width)*0.12);
			tempbox.width = r->width;
			tempbox.height = r->height;
			facebox.push_back(tempbox);

			}

        }  
        cvReleaseMemStorage(&pcvMemStorage);  
    }  

	return facebox;

}

//对灰度图水平镜像
void hMirrorTrans(const Mat &src, Mat &dst)
{
    dst.create(src.rows, src.cols, src.type());

    int cols = src.cols;

    for (int i = 0; i < cols; i++)
        src.col(cols - i - 1).copyTo(dst.col(i));
}