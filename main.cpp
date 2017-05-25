#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
using namespace std;
using namespace cv;

String face_cascade = "/home/ubuntu/code/haarcascade_frontalface_alt.xml";
String eye_cascade = "/home/ubuntu/code/haarcascade_eye.xml";
String img_name = "/home/ubuntu/photo/ljs.png";

CascadeClassifier face;
CascadeClassifier eye;
vector < vector < Point > > contours;
double sensitiveArea,face_area;
bool findPimples(Mat img)
{
    Mat bw, bgr[3];
    split(img, bgr);
    bw = bgr[1];
    
    adaptiveThreshold(bw, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, 3);
    dilate(bw, bw, Mat(), Point(-1, -1), 1);
    contours.clear();
    findContours(bw, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    
    for (size_t i = 0; i< contours.size(); i++){
        
        if (contourArea(contours[i]) > 20 && contourArea(contours[i]) < 400)
        {
            Rect minRect = boundingRect(Mat(contours[i]));
            Mat imgroi(img, minRect);
            
            cvtColor(imgroi, imgroi, COLOR_BGR2HSV);
            cvtColor(imgroi, imgroi, COLOR_HSV2BGR);
            Scalar b_color = mean(imgroi);
            
            bool black= b_color[0] < 20 && b_color[1] < 20 && b_color[2] < 20;
            
            if (!black) {
                Point2f center;
                float radius = 0;
                minEnclosingCircle(Mat(contours[i]), center, radius);
                if (radius < 25) {
                    sensitiveArea += contourArea(contours[i]);
                }
            }
        }
    }
    return true;
}
Mat mkKernel(int ks, double sig, double th, double lm, double ps)
{
    int hks = (ks - 1) / 2;
    double theta = th*CV_PI / 180;
    double psi = ps*CV_PI / 180;
    double del = 2.0 / (ks - 1);
    double lmbd = lm;
    double sigma = sig / ks;
    double x_theta;
    double y_theta;
    Mat kernel(ks, ks, CV_32F);
    for (int y = -hks; y <= hks; y++)
    {
        for (int x = -hks; x <= hks; x++)
        {
            x_theta = x*del*cos(theta) + y*del*sin(theta);
            y_theta = -x*del*sin(theta) + y*del*cos(theta);
            kernel.at<float>(hks + y, hks + x) = (float)exp(-0.5*(pow(x_theta, 2) + pow(y_theta, 2)) / pow(sigma, 2))* cos(2 * CV_PI*x_theta / lmbd + psi);
        }
    }
    return kernel;
}
int kernel_size = 21;
int pos_sigma = 4;
int pos_lm = 50;
int pos_th = 0;
int pos_psi = 96;
cv::Mat src_f;
cv::Mat dest;
int Process(int, void *)
{
    double sig = pos_sigma;
    double lm = 0.5 + pos_lm / 100.0;
    double th = pos_th;
    double ps = pos_psi;
    Mat kernel = mkKernel(kernel_size, sig, th, lm, ps);
    filter2D(src_f, dest, CV_32F, kernel);
    Mat Lkernel(kernel_size * 20, kernel_size * 20, CV_32F);
    resize(kernel, Lkernel, Lkernel.size());
    Lkernel /= 2.;
    Lkernel += 0.5;
    Mat mag;
    pow(dest, 2.0, mag);
    erode(mag, mag, Mat());
    
    Mat opening;
    medianBlur(mag, opening, 5);
    
    Mat diff;
    absdiff(mag, opening, diff);
    Mat thimg = diff.clone();
    
    thimg.convertTo(thimg, CV_8UC1, 255.0);
    cvtColor(thimg, thimg, CV_GRAY2BGR);
    vector<Mat> bgr_images(3);
    split(thimg, bgr_images);
    
    int wrinklecnt = 0;
    for(int row=0;row<thimg.rows;row++) {
        for(int col=0;col < thimg.cols;col++) {
            int B = thimg.at<Vec3b>(row, col)[0];
            int G = thimg.at<Vec3b>(row, col)[1];
            int R = thimg.at<Vec3b>(row, col)[2];
            if(B<=30 && G<=30 && R<=30) {
                thimg.at<Vec3b>(row, col)[0]=0;
                thimg.at<Vec3b>(row, col)[1]=0;
                thimg.at<Vec3b>(row, col)[2]=0;
            }
            else {
                wrinklecnt++;
            }
        }
    }
    
    wrinklecnt = (wrinklecnt)*1000 / (thimg.rows*thimg.cols);
    return wrinklecnt;
}

int main() {
    Mat img;
    img = imread(img_name);
    if (!img.data) {
        puts("cannot");
        return -1;
    }
    if (!face.load(face_cascade)) {
        puts("cascade face fail");
        return -1;
    }
    if (!eye.load(eye_cascade)) {
        puts("cascade eye fail");
        return -1;
    }
    Mat gray;
    cvtColor(img, gray, CV_RGB2GRAY);
    vector<Rect> face_pos;
    equalizeHist(gray, gray);
    face.detectMultiScale(gray, face_pos, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    
    if(face_pos.size() == 0) {
      //  puts("face detect fail");
        puts("-1\n-1\n-1");
        return -1;
    }
    vector<Rect> eye_pos;
    Mat roi = gray(face_pos[0]);
    Mat eyeROI = img(Rect(face_pos[0].x, face_pos[0].y, face_pos[0].width, face_pos[0].height / 2));
    cvtColor(eyeROI, roi, CV_RGB2GRAY);
    equalizeHist(roi, roi);
    
    eye.detectMultiScale(roi, eye_pos, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(40, 40));
    
    if(eye_pos.size() == 0) {
      //  puts("eye detect fail");
        puts("-1\n-1\n-1");
        return -1;
    }
    
    cvtColor(img, img, CV_BGR2HLS);
    vector<Mat> hls_images(3);
    split(img, hls_images);
    
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            uchar H = img.at<Vec3b>(row, col)[0];
            uchar L = img.at<Vec3b>(row, col)[1];
            uchar S = img.at<Vec3b>(row, col)[2];
            
            double LS_ratio = ((double)L) / ((double)S);
            
            bool skin_pixel = (H < 18) && (S >= 50) && (LS_ratio > 0.5) && (LS_ratio < 3.0);
            bool face_pixel = (col >= face_pos[0].x) && (col< face_pos[0].x+ face_pos[0].width) && (row >= face_pos[0].y) && (row< face_pos[0].y + face_pos[0].height);
            
            if (skin_pixel == false || face_pixel == false) {
                img.at<Vec3b>(row, col)[0] = 0;
                img.at<Vec3b>(row, col)[1] = 0;
                img.at<Vec3b>(row, col)[2] = 0;
            }
        }
    }
    cvtColor(img, img, CV_HLS2BGR);
    
    erode(img, img, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);
    erode(img, img, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);
    erode(img, img, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);
    
    Mat underEye = img(Rect(face_pos[0].x, face_pos[0].y+eye_pos[0].y+eye_pos[0].height, face_pos[0].width, face_pos[0].height - (eye_pos[0].y+eye_pos[0].height)));
    
    cvtColor(underEye, underEye, CV_BGR2HLS);
    split(underEye, hls_images);
    
    for (int row = 0; row < underEye.rows; row++) {
        for (int col = 0; col < underEye.cols; col++) {
            uchar H = underEye.at<Vec3b>(row, col)[0];
            uchar L = underEye.at<Vec3b>(row, col)[1];
            uchar S = underEye.at<Vec3b>(row, col)[2];
            
            if(!(H <= 10 && L<=10 && S<= 10))
                face_area += 1;
        }
    }
    cvtColor(underEye, underEye, CV_HLS2BGR);
    findPimples(underEye);
    double pimpleratio = sensitiveArea*100/face_area;
    
 //   printf("pimpleratio : %f\n",sensitiveArea*100/face_area);
    
    Mat wrinkle = img(Rect(face_pos[0].x, face_pos[0].y, face_pos[0].width, face_pos[0].height));
    Mat src;
    cvtColor(wrinkle, src, CV_BGR2GRAY);
    src.convertTo(src_f, CV_32F, 1.0 / 255, 0);
    
    if (!kernel_size % 2)
        kernel_size += 1;
    
    int wrinkleratio = Process(0, 0);
//    printf("wrinkle : %f\n",wrinkleratio); // wrinkle
    
    int ystart = face_pos[0].y + eye_pos[0].y + eye_pos[0].height + face_pos[0].height/16;
    int xstart = face_pos[0].x + eye_pos[0].x + eye_pos[0].width/2;
    int ysize = ystart + face_pos[0].height/8;
    int xsize = xstart + eye_pos[0].width/2;
    
    
    cvtColor(img, img, CV_BGR2HLS);
    
    split(img, hls_images);
    double resLS_ratio = 0;
    double rescnt = 0;
    for (int row = ystart; row < ysize; row++) {
        for (int col = xstart; col < xsize; col++) {
            int L = img.at<Vec3b>(row, col)[1];
            int S = img.at<Vec3b>(row, col)[2];
            double LS_ratio = ((double)L) / ((double)S);
            resLS_ratio+=LS_ratio;
            rescnt += 1;
        }
    }
    resLS_ratio = resLS_ratio/rescnt; // Skin tone
    // bright
  //  printf("skin tone : ");
    if(resLS_ratio <= 1.4) {
        puts("0");
    }
    else if(resLS_ratio >1.4 && resLS_ratio<=1.8) { // usually
        puts("1");
    }
    else { //gloomy
        puts("2");
    }
    cvtColor(img, img, CV_HLS2BGR);
    
    if(pimpleratio > 1.4)
        puts("0");
    
    else
        puts("1");
    
    printf("%d\n",wrinkleratio);
    
    
    waitKey();
    
    
    return 0;
}


