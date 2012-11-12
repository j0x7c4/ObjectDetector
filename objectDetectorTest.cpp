#include <cv.h>  
#include <highgui.h> 
#include <ml.h>  
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>  
using namespace cv;  
using namespace std;  
  
  
int main(int argc, char** argv)    
{    
    vector<string> img_path;  
    vector<int> img_catg;  
    int nLine = 0;  
    string buf;  
    ifstream svm_data( "E:/source/gestureRecognition/objectDetector/svm_data" );  
    unsigned long n;  
  
    while( svm_data )  
    {  
        if( getline( svm_data, buf ) )  
        {   
            if( nLine % 2 == 0 )  
            {  
                 img_catg.push_back( atoi( buf.c_str() ) );//atoi将字符串转换成整型，标志（0,1）  
            }  
            else  
            {  
                img_path.push_back( buf );//图像路径  
            }
						nLine ++; 
        }  
    }  
    svm_data.close();//关闭文件  
  
    CvMat *data_mat, *res_mat;  
    int nImgNum = nLine / 2;            //读入样本数量  
    //////样本矩阵，nImgNum：横坐标是样本数量， WIDTH * HEIGHT：样本特征向量，即图像大小  
    data_mat = cvCreateMat( nImgNum, 1764, CV_32FC1 );  
    //cvSetZero( data_mat );  
    ////类型矩阵,存储每个样本的类型标志  
    res_mat = cvCreateMat( nImgNum, 1, CV_32FC1 );  
    //cvSetZero( res_mat );  
  
    IplImage* src;  
    IplImage* trainImg=cvCreateImage(cvSize(64,64),8,3);//需要分析的图片  
  
    for( string::size_type i = 0; i != img_path.size(); i++ )  
    {  
            src=cvLoadImage(img_path[i].c_str(),1);  
            if( src == NULL )  
            {  
                cout<<" can not load the image: "<<img_path[i].c_str()<<endl;  
                continue;  
            }  
  
            cout<<" processing "<<img_path[i].c_str()<<endl;  
                 
            cvResize(src,trainImg);   //读取图片     
            HOGDescriptor *hog=new HOGDescriptor(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);  //具体意思见参考文章1,2     
            vector<float>descriptors;//结果数组     
            hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //调用计算函数开始计算     
            cout<<"HOG dims: "<<descriptors.size()<<endl;  
            //CvMat* SVMtrainMat=cvCreateMat(descriptors.size(),1,CV_32FC1);  
            n=0;  
            for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)  
            {  
                cvmSet(data_mat,i,n,*iter);  
                n++;  
            }  
                //cout<<SVMtrainMat->rows<<endl;  
            cvmSet( res_mat, i, 0, img_catg[i] );  
            cout<<" end processing "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;  
    }  
      
               
    CvSVM svm = CvSVM();    
    CvSVMParams param;    
    CvTermCriteria criteria;    
    criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );    
    param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );    
/*    
    SVM种类：CvSVM::C_SVC    
    Kernel的种类：CvSVM::RBF    
    degree：10.0（此次不使用）    
    gamma：8.0    
    coef0：1.0（此次不使用）    
    C：10.0    
    nu：0.5（此次不使用）    
    p：0.1（此次不使用）    
    然后对训练数据正规化处理，并放在CvMat型的数组里。    
                                                        */       
    //☆☆☆☆☆☆☆☆☆(5)SVM学习☆☆☆☆☆☆☆☆☆☆☆☆         
    //svm.train( data_mat, res_mat, NULL, NULL, param );    
    //☆☆利用训练数据和确定的学习参数,进行SVM学习☆☆☆☆     
    //svm.save( "SVM_DATA.xml" );    
    //检测样本  
		svm.load("SVM_DATA.xml");
    IplImage *test;  
    vector<string> img_tst_path;  
    ifstream img_tst( "E:/source/gestureRecognition/objectDetector/svm_test" );  
    while( img_tst )  
    {  
        if( getline( img_tst, buf ) )  
        {  
            img_tst_path.push_back( buf );  
        }  
    }  
    img_tst.close();  
  
		
  
    CvMat *test_hog = cvCreateMat( 1, 1764, CV_32FC1 );  
    char line[512];  
    ofstream predict_txt( "SVM_PREDICT.txt" );  
    for( string::size_type j = 0; j != img_tst_path.size(); j++ )  
    {  
        test = cvLoadImage( img_tst_path[j].c_str(), 1);  
        if( test == NULL )  
        {  
             cout<<" can not load the image: "<<img_tst_path[j].c_str()<<endl;  
               continue;  
         }  
          
        cvZero(trainImg);  
        cvResize(test,trainImg);   //读取图片     
        HOGDescriptor *hog=new HOGDescriptor(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);  //具体意思见参考文章1,2     
        vector<float>descriptors;//结果数组     
        hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //调用计算函数开始计算     
        cout<<"HOG dims: "<<descriptors.size()<<endl;  
        CvMat* SVMtrainMat=cvCreateMat(1,descriptors.size(),CV_32FC1);  
        n=0;  
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)  
            {  
                cvmSet(SVMtrainMat,0,n,*iter);  
                n++;  
            }  
  
        int ret = svm.predict(SVMtrainMat);  
        sprintf( line, "%s %d\r\n", img_tst_path[j].c_str(), ret );  
         predict_txt<<line;  
    }  
    predict_txt.close();  
  
//cvReleaseImage( &src);  
//cvReleaseImage( &sampleImg );  
//cvReleaseImage( &tst );  
//cvReleaseImage( &tst_tmp );  
cvReleaseMat( &data_mat );  
cvReleaseMat( &res_mat );  
  
return 0;  
}  