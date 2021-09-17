#include <stdio.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <math.h>       /* round, floor, ceil, trunc */

using namespace cv;
using namespace std;

// This will be the hand written function for convolution of an image
Mat convolution (Mat base_img, Mat kernel);
Mat houghCircles(Mat grad, Mat img);
Mat direction (Mat xs, Mat ys);
int ***malloc3dArray(int dim1, int dim2, int dim3);

int main(){

  // read img
  Mat img = imread("dart1.jpg", 1);
  Mat image,img2;
  img2 = img.clone();
  imwrite("image2.jpg", img2);
  cvtColor( img, img, CV_BGR2GRAY );
  GaussianBlur(img, image, Size(7 ,7 ),0 ,0 );
  // set up tranform kernel
  Mat xKernel = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
  Mat yKernel = (Mat_<float>(3,3) << -1,-2,-1,0,0,0,1,2,1);
  Mat ys = convolution(image, yKernel);
  Mat xs = convolution(image, xKernel);
  printf("finished conv\n");
  Mat grad;
  grad = xs.mul(xs) + ys.mul(ys);
  sqrt(grad, grad);
  Mat norm = grad.clone();
  normalize(grad, grad, 0, 255, NORM_MINMAX);
  Mat angle = direction(xs, ys);
  Mat canny ;
  Canny(img, canny, 110, 380, 3);
  Mat hough = houghCircles(canny,img2);

  xs.convertTo(xs,CV_8UC1);
  ys.convertTo(ys,CV_8UC1);
  norm.convertTo(norm,CV_8UC1);
  grad.convertTo(grad,CV_8UC1);
  angle.convertTo(angle,CV_8UC1);
  canny.convertTo(canny,CV_8UC1);
  hough.convertTo(hough,CV_8UC1);
  // Now start working on Hough transformation
  imshow("Canny", canny);
  waitKey(0);
  imshow("hough", hough);
  waitKey(0);

/*
  //construct a window for image display
  //namedWindow("Display window", WINDOW_AUTOSIZE);
  //visualise the loaded image in the window
  imshow("X_derivative", xs);
  waitKey(0);//wait for a key press until returning from the program
  imshow("Y_derivative", ys);
  waitKey(0);
  imshow("Before Normalization", norm);
  waitKey(0);
  imshow("Gradient Magnitude", grad);
  waitKey(0);
  imshow("Canny", canny);
  waitKey(0);
  imshow("Gradient angle", ancannygle);
  waitKey(0);
*/
  imwrite("X_derivative.jpg", xs);
  imwrite("Y_derivative.jpg", ys);
  imwrite("Before Normalization.jpg", norm);
  imwrite("Gradient Magnitude.jpg", grad);
  imwrite("GBlur.jpg", image);
  imwrite("Degrees.jpg", angle);
  imwrite("Canny.jpg", canny);
  imwrite("Hough.jpg", hough);

  //free memory occupied by image
  image.release();
  xKernel.release();
  yKernel.release();
  xs.release();
  ys.release();
  grad.release();
  angle.release();
  canny.release();
  hough.release();
  return 0;
}


Mat houghCircles(Mat grad, Mat img){

  Mat img_h = img.clone();

  int a, b, T = 130 ;
  int rmin = 35, rmax = 145;
  int ***acc;
  acc = malloc3dArray(grad.rows, grad.cols, rmax);

  float pix ;

  //float acc [500][500][rmax];
  //cout << grad.rows << endl;
  //cout << grad.cols << endl;
  for(int x = 0 ; x < grad.rows ; x++)
  {
    for(int y = 0 ; y < grad.cols ; y++)
    {
      for(int r = 0 ; r < rmax ; r++)
      {
        acc[x][y][r] = 0;
      }
    }
  }

  for(int x = 0 ; x < grad.rows ; x++)
  {
    for(int y = 0 ; y < grad.cols ; y++)
    {
      pix = grad.at<float>(x,y);
      if (pix > T)
      {
        for(int r = rmin ; r < rmax ; r++)
        {
          for(int t = 0 ; t < 360 ; t ++)
          {
            a = (int)(x - r * cos((double)(t * M_PI / 180)));
            b = (int)(y - r * sin((double)(t * M_PI / 180)));
            //voting
            if(a>0 && a<grad.rows && b>0 && b<grad.cols){  acc[a][b][r] += 1;}
          }
        }
      }
    }
  }

  int thresh = 10;
  for(int x = 0 ; x < grad.rows ; x++)
  {
    for(int y = 0 ; y < grad.cols ; y++)
    {
      for(int r = rmin ; r < rmax ; r++)
      {

        if(acc[x][y][r] > 10){
          circle(img_h, Point(x,y), 2, cvScalar(0,0,255), 2);
          cout << x<<" "<<y<<" "<<" "<<r<<" "<<acc[x][y][r] << endl;
        }
      }
    }
  }
  return img_h;

}

int ***malloc3dArray(int dim1, int dim2, int dim3)
{
    int i, j, k;
    int **array = (int **) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
	for (j = 0; j < dim2; j++) {
  	    array[i][j] = (int *) malloc(dim3 * sizeof(int));
	}

    }
    return array;
}

Mat direction (Mat xs, Mat ys){

  Mat theta = xs.clone();
  float dx,dy, ang;

  for (int x = 0; x < xs.rows; x++){
    for (int y = 0; y < xs.cols; y++){
      dx = xs.at<float>(x,y);
      dy = ys.at<float>(x,y);
      //ang = atan(dy/dx)* 180 / M_PI;
      theta.at<float>(x,y) = fastAtan2(dy,dx);// in radians
    }
  }
  return theta;
}

Mat convolution(Mat base_img, Mat kernel){

  Mat new_img = base_img.clone(); //create output image
  new_img.convertTo(new_img, CV_32F); //convert to float so we can get larger numbers

  //Loop through the image
  for (int y = 0; y < base_img.rows; y++){
    for (int x = 0; x < base_img.cols; x++){

      //reset sum at the start of a new pixel
      float sum = 0;

      //Loop through the kernel
      for (int i = -1; i <= 1; i++){
        for (int j = -1; j <= 1; j++){
          //check if you go out of image range
          if (!(y+i < 0 || y+i > base_img.rows || x+j < 0 || x+j > base_img.cols)){
            sum += (base_img.at<uchar>(y+i,x+j) * kernel.at<float>(i+1,j+1));
          }
        }
      }

      //set the sum to the new image
      new_img.at<float>(y,x) = sum;
    }
  }

  //Mat norm_img;
  //Norm image
  //normalize(new_img, new_img, 0, 255, NORM_MINMAX);

  return new_img;
}
