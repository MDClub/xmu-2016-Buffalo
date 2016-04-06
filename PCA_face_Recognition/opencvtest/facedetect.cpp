#include "opencvtest.h"

using namespace std;  
using namespace cv; 


//��ⵥ֡ͼ�������������
/*
����˵��
pSrcImage:��֡ͼ��
pHaarClassCascade��OpenCV�Դ�����Haar���������
min_neighbor���ж��Ƿ�Ϊ�����ı�׼��ȡֵ1��2...��Խ�󣬱����������������Ŀ�����Խ��
minsize����Ҫ������С�����ߴ�
templet��true��ʾ�Ƕ�ģ��ͼƬ���м�⣬flase��ʾ�Ƕ���ƵͼƬ���м��
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
    cvCvtColor(pSrcImage, pGrayImage, CV_BGR2GRAY); //תΪ�Ҷ�ͼ 
  
    if (pHaarClassCascade != NULL && pSrcImage != NULL && pGrayImage != NULL)  
    {          
        CvMemStorage *pcvMemStorage = cvCreateMemStorage(0);  
        cvClearMemStorage(pcvMemStorage);  

        //detect the face
        int TimeStart, TimeEnd;  
        TimeStart = GetTickCount();  //������ʱ��
        CvSeq *pcvSeqFaces = cvHaarDetectObjects(pGrayImage, pHaarClassCascade, pcvMemStorage, 1.1, min_neighbor, CV_HAAR_DO_CANNY_PRUNING,minsize);  
        TimeEnd = GetTickCount();  
  
		#ifdef DEBUG
        printf("Spending Time: %d ms\n", TimeEnd - TimeStart);//ÿ֡ͼ����������ʱ��
        #endif

        //mark the face   
        for(int i = 0; i <pcvSeqFaces->total; i++)  
        {  
            CvRect* r = (CvRect*)cvGetSeqElem(pcvSeqFaces, i);  

			//��⵽�ľ��ο��ϡ����ұ߿�����������5%
            Rect tempbox;
			tempbox.x = r->x + floor((r->width)*0.05);
			tempbox.y = r->y + floor((r->width)*0.05);
			tempbox.width = r->width - 2*floor((r->width)*0.05);
			tempbox.height = r->height - 2*floor((r->width)*0.05);
			facebox.push_back(tempbox);

			//�Լ�⵽��ͼƬ���ж෽λλ�õ���
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

//�ԻҶ�ͼˮƽ����
void hMirrorTrans(const Mat &src, Mat &dst)
{
    dst.create(src.rows, src.cols, src.type());

    int cols = src.cols;

    for (int i = 0; i < cols; i++)
        src.col(cols - i - 1).copyTo(dst.col(i));
}