
/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

int ***malloc3dArray(int dim1, int dim2, int dim3);

Mat threshold(Mat grad);
Mat gradient(Mat image);
Mat direction (Mat image);
Mat convolution (Mat base_img, Mat kernel);
void groundVals(vector<Rect> faces, vector<Rect>ground);
Mat houghLines( Mat img );
Mat houghCircles( Mat img);



/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
vector<Rect> circleVec;							//circles from hough above threshold
vector<Rect> detect;								//final detections
vector<Rect> ground; 								//ground truths
vector<Rect> strongestCircle;				//strongest circle from hough
vector<Point> intersectionVec;		 	//lines from hough above threshold
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{

       	// 1. Read Input Image
	Mat frame = imread(argv[1], 1);
				// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// Add in ground truths
	//ground.push_back(Rect(420,0,210,220));   //img0
	 //ground.push_back(Rect(175,100,235,230));   //img1
	//ground.push_back(Rect(90,85,110,110));     //img2
 	//ground.push_back(Rect(310,135,90,95));		 //img3
	 //ground.push_back(Rect(155,65,380-155,335-65)); //img4
	 //ground.push_back(Rect(415,125,550-415,270-125));	//img 5
	 //ground.push_back(Rect(205,110,280-205,190-110));	//img 6
	 //ground.push_back(Rect(230,150,410-230,335-140));	//img 7
	 //ground.push_back(Rect(830,200,970-830,350-200));	//img 8
	 //ground.push_back(Rect(60,240,130-60,350-240));	//img 8
	 //ground.push_back(Rect(170,15,465-170,320-15));	//img 9
	//ground.push_back(Rect(80,90,195-80,230-90));	//img 10
	 //ground.push_back(Rect(575,120,640-575,220-120));	//img 10
	 //ground.push_back(Rect(914,140,955-914,220-130));	//img 10
 	ground.push_back(Rect(165,95,240-165,180-95));	//img 11
	 //ground.push_back(Rect(150,60,220-150,230-60));	//img 12
	// ground.push_back(Rect(255,100,420-255,270-100));	//img 13
	//ground.push_back(Rect(105,85,260-105,240-85));	//img 14
	//ground.push_back(Rect(970,80,1125-970,235-80));	//img 14
	 //ground.push_back(Rect(120,35,300-120,210-35));	//img 15

	Mat img = frame.clone();
	Mat img2=img.clone();

				// 3. Hough transforms- circles & lines
  Mat hcircle = houghCircles( img2 );
	Mat hline = houghLines( img2 );


				// 4. Detect Faces and Display Result
	detectAndDisplay( frame );
	groundVals(detect,ground);


	// PLOTTING DETECTION BY CODE
	for (int i = 0; i < detect.size(); i++){
		int x = detect[i].x;	int y = detect[i].y;
		rectangle( frame , Point(x,y),Point(x + detect[i].width,y + detect[i].height),Scalar( 0, 255, 0 ),2);
	}

	// PLOTTING ACTUAL DARTBOARD- GROUND TRUTH
	for (int i = 0; i < ground.size(); i++){
		int x = ground[i].x;	int y = ground[i].y;
		rectangle( frame , Point(x,y),Point(x + ground[i].width,y + ground[i].height),Scalar( 0, 0, 255 ),2);
	}

	//rectangle( frame , Point(x,y),Point(x + circleVec[i].width,y + circleVec[i].height),Scalar( 30, 30, 255 ),2);


	// 4. Save Result Image
	imshow("Final conclusion", frame);
	imwrite( "detected.jpg", frame );
	waitKey(0);

	return 0;
}

// This function takes all the faces found by viola-jones, and does IOU with the ground truths
// it then returns the IOU value as a float
void groundVals(vector<Rect> faces, vector<Rect>ground){
	vector<float> IOUs;	//store the IOU values of a face compared to all the truths

	vector<float> bestIOUs;		//store the best result for a give face vs truth
	float maxArea=0.0,Area;
	int indexT,indexF;

	for ( int j = 0; j < faces.size(); j++ ){	//loop through the truth boundries
		IOUs.clear();
		for( int i = 0; i < ground.size(); i++ )		// loop through the generated boundries
		{
			Rect inter = faces[j] & ground[i];	//get intersection
			Rect unions = faces[j] | ground[i];	//get union
			//printf("Inter area: %d.  Union area: %d.   IOU: %f\n", inter.area(), unions.area(), (float)inter.area()/(float)unions.area());
			Area = (float)inter.area() / (float)unions.area();
			if(Area> maxArea)
			{
				maxArea = Area;
				indexT = i ;//index of ground[i]
				indexF = j ;//index of faces[j]
			}

			IOUs.push_back(Area);
	 	}

		bestIOUs.push_back(*max_element(IOUs.begin(),IOUs.end()));
	}

	int TP = 0;	//count true positives
	int FP = 0;	//count flase positives
	//get recall (TPR)
	for (int i = 0; i < bestIOUs.size(); i++){
		if (bestIOUs[i] > 0.4){	//if UOI val is > 0.5 count as true positive

			detect.push_back(Rect(faces[i].x,faces[i].y,faces[i].width,faces[i].width));
			TP++;
			printf("True P found @ (%d,%d,%d,%d): %f accuracy\n", bestIOUs[i],faces[i].x,faces[i].y,faces[i].width,faces[i].width);
		}else {
			FP++;
			printf("False P found @ (%d,%d,%d,%d): %f accuracy\n", bestIOUs[i],faces[i].x,faces[i].y,faces[i].width,faces[i].height);
		}

	}

	float recall = (float)TP/ground.size();
	float prec	= (float)TP / ((float)TP + (float)FP);

	float FOne = 2.0f * ((prec * recall) / (prec + recall));

	printf("Recall [TPR] = %f\n",recall);
	printf("precision = %f\n",prec);
	printf("F1 score = %f\n",FOne);
}


// This function combines the detection of viola-jones, hough circles and hough lines
void iouVal(vector<Rect> faces){
	vector<float> IOUs;	//store the IOU values of a face compared to all the circleVec

	vector<float> bestIOUs;		//store the best result for a given face vs circleVec
	float maxArea=0.0,Area;
	int indexT,indexF;

	if(circleVec.empty())
	{
		circleVec.push_back(Rect(strongestCircle[0].x,strongestCircle[0].y,strongestCircle[0].width,strongestCircle[0].width));
	}

	for ( int j = 0; j < faces.size(); j++ ){	//loop through the viola-jones detection
		IOUs.clear();
		for( int i = 0; i < circleVec.size(); i++ )		// loop through the circle detection
		{
			Rect inter = faces[j] & circleVec[i];	//get intersection
			Rect unions = faces[j] | circleVec[i];	//get union
			//printf("Inter area: %d.  Union area: %d.   IOU: %f\n", inter.area(), unions.area(), (float)inter.area()/(float)unions.area());
			Area = (float)inter.area() / (float)unions.area();
			if(Area> maxArea)
			{
				maxArea = Area;
				indexT = i ;//index of circleVec[i]
				indexF = j ;//index of faces[j]
			}

			IOUs.push_back(Area);
	 	}

		bestIOUs.push_back(*max_element(IOUs.begin(),IOUs.end()));
	}

	// if(bestIOUs.empty())
	// {
	// 	for ( int j = 0; j < faces.size(); j++ ){	//loop through the viola-jones detection
	// 		IOUs.clear();
	// 		for( int i = 0; i < intersectionVec.size(); i++ )		// loop through the circle detection
	// 		{
	// 			int x1 = intersectionVec[i].x;
	// 			int y1 = intersectionVec[i].y;
	// 			if(x1 >= faces[j].x && x1<= faces[j].x+faces[j].width && y1 >=faces[j].y && y1 <= faces[j].y+faces[j].width){
	// 				detect.push_back(Rect(faces[j].x,faces[j].y,faces[j].width,faces[j].width));cout<<"fff";
	// 			}
	// 		}
	// 	}
	// }


	int TP = 0;	//count true positives
	int FP = 0;	//count flase positives
	//get recall (TPR)
	for (int i = 0; i < bestIOUs.size(); i++){
		if (bestIOUs[i] > 0.8){	//if IOU val is > 0.5 count as true positive

			detect.push_back(Rect(faces[i].x,faces[i].y,faces[i].width,faces[i].width));
			//TP++;
			//printf("True P found @ (%d,%d,%d,%d): %f accuracy\n", bestIOUs[i],detect[i].x,detect[i].y,detect[i].width,detect[i].width);
		}
		// else {
		// 	FP++;
		// 	printf("False P found @ (%d,%d,%d,%d): %f accuracy\n", bestIOUs[i],faces[i].x,faces[i].y,faces[i].width,faces[i].height);
		// }



	}
	/*
	float recall = (float)TP/circleVec.size();
	float prec	= (float)TP / ((float)TP + (float)FP);

	float FOne = 2.0f * ((prec * recall) / (prec + recall));

	printf("Recall [TPR] = %f\n",recall);
	printf("precision = %f\n",prec);
	printf("F1 score = %f\n",FOne);*/




}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CASCADE_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	//std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	// for( int i = 0; i < faces.size(); i++ )
	// {
	// 	rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	// }

	iouVal(faces);


}


//*************************************************************************************************************************************************************

Mat houghLines( Mat img )
{
	Mat img_h = img.clone();
  Mat gray;
  cvtColor( img_h, gray, CV_BGR2GRAY );
  //GaussianBlur(gray, gray, Size(9 ,9 ),0 ,0 );
	//equalizeHist( gray, gray );
  Mat grad =  gradient(gray);   //pixel range 0-255
  Mat dir = direction(gray);    //in radians
  grad = threshold(grad);       //pix-  either 0 or 255
  int diagonal = sqrt((grad.rows*grad.rows)+(grad.cols*grad.cols));
  Mat houghSpace (diagonal*2,180*2, CV_32FC1, Scalar(0)) ;
  Mat intersectionSpace (grad.rows,grad.cols, CV_32FC1, Scalar(0)) ;
  //cout<< endl << "begin lines" << endl;



  //VOTING in houghSpace
  int P;int degrees = 0;
  for(int y = 0 ; y < grad.rows ; y++)
  {
    for(int x = 0 ; x < grad.cols ; x++)
    {
      if(grad.at<float>(y,x) == 255)
      {
        for(int t= 0 ; t< (180)*1; t++)
        {
          float radians = t*  (M_PI/180);
          P =  ( x*cos(radians) + y*sin(radians) ) + (diagonal);
          houghSpace.at<float> (P,t) += 1;
        }
      }
    }
  }
  //cout<< endl << "Start hough space loop" <<endl;



  //THRESHOLDING rho & theta pairs
  std::vector<Point> pair;
  int voteTh = 100; int maxPt = 0;
  for(int j=0 ; j < houghSpace.rows ; j++)
  {
    for(int i=0 ; i < houghSpace.cols ; i++)
    {
      maxPt = (houghSpace.at<float>(j,i)>maxPt)? houghSpace.at<float>(j,i) : maxPt ;
      if (houghSpace.at<float>(j,i) > voteTh){
        pair.push_back(Point(j- diagonal,i));//cout << Point(j,i) << "! ";
      }
    }
  }
  cout <<endl<< "Max points in a line : " << maxPt;
  //cout << "\n\nEnd 2D Hough Space - Start intersection Space\n";



  //PLOTTING lines
  int x1,x2,y1,y2;
  int rho;float theta;
  for(int index = 0 ; index < pair.size() ; index++)
  {
    rho = pair[index].x ;
    theta = pair[index].y * (M_PI/180);
    //cout << endl << "rho = "<< rho << " , theta = " << theta << endl ;
    y1 = rho*sin(theta) + 1000 * cos(theta);
    y2 = rho*sin(theta) - 1000 * cos(theta);
    x1 = rho*cos(theta) - 1000 * sin(theta);
    x2 = rho*cos(theta) + 1000 * sin(theta);
    line( img_h, Point(x1,y1), Point(x2, y2), cvScalar(0,0,255),3);

    //VOTING intersection in Mat intersectionSpace
    for(int x = 0 ; x < intersectionSpace.cols ; x++){
        int y =    ( ( rho / sin(theta) - (x) / tan(theta) ) )  ;
        if(y >= 0 && y < intersectionSpace.rows){
          intersectionSpace.at<float>(y,x) +=50;//cout << Point(y,x) << "! ";
        }
    }
  }


  //Finding MAX INTERSECTION pt
  int maxIn = 0,maxi_Y = 0, maxi_X = 0;
	int interTH = 1;
  for(int y = 0 ; y < intersectionSpace.rows ; y++){
    for(int x = 0 ; x < intersectionSpace.cols ; x++){
      if(intersectionSpace.at<float>(y,x) > maxIn){
        maxIn = intersectionSpace.at<float>(y,x);
        maxi_Y = y;maxi_X = x;
      }
			if(intersectionSpace.at<float>(y,x)/50 > interTH){
				intersectionVec.push_back(Point(x,y));
				circle(img_h, Point(x,y), 2, cvScalar(255,0,0), 2);
			}
    }
  }


  cout <<endl<< "LINES - Max intersections between lines : " << maxIn/50 << " at " << Point(maxi_Y,maxi_X) << endl;
	circle(img_h, Point(maxi_X,maxi_Y), 2, cvScalar(0,255,0), 2);
  //*************************************************************************************************************************************************************
	normalize(houghSpace, houghSpace, 0, 255, NORM_MINMAX);houghSpace.convertTo(houghSpace,CV_8UC1);
  //imshow("Hough Space LINES ", houghSpace);
  imwrite("Hough Space LINES.jpg", houghSpace);
  //normalize(intersectionSpace, intersectionSpace, 0, 255, NORM_MINMAX);
  intersectionSpace.convertTo(intersectionSpace,CV_8UC1);
  //cout << intersectionSpace << '\n';
  //imshow("Intersection Space LINES ", intersectionSpace);
  imwrite("Intersection Space LINES.jpg", intersectionSpace);
  imshow("hough Lines", img_h);
  imwrite("hough Lines.jpg", img_h);

  return img_h;
}

//*************************************************************************************************************************************************************


Mat houghCircles(Mat img){

	Mat img_h = img.clone();
	Mat gray;
  cvtColor( img_h, gray, CV_BGR2GRAY );
	//GaussianBlur(gray, gray, Size(9 ,9 ),0 ,0 );
	//equalizeHist( gray, gray );
  Mat grad = gradient(gray);      //pixel range 0-255
  Mat dir = direction(gray);    //in radians
  grad = threshold(grad);   //pixel either 0 or 255

  // Voting Threshold * * *
  int voteTh = 7 ;

  // Radius Range * * *
  int rmin = 40, rmax = 140;

  // declaring accumulator
  int ***acc; acc = malloc3dArray(grad.rows, grad.cols, rmax);

  int a,b,c,d;

  //INTILIALISING accumulator
  for(int y = 0 ; y < grad.rows ; y++){
    for(int x = 0 ; x < grad.cols ; x++){
      for(int r = 0 ; r < rmax ; r++){
        acc[y][x][r] = 0;}}}

  //---------------------------------------------------------------
  //VOTING
  float angle = 0;
  for(int y = 0 ; y < grad.rows ; y++)
  {
    for(int x = 0 ; x < grad.cols ; x++)
    {
      if (grad.at<float>(y,x) == 255)
      {
        angle = dir.at<float>(y,x);
        for(int r = rmin ; r < rmax ; r++)
        {
          a = (int)(y - r * sin(angle));
          b = (int)(x - r * cos(angle));//cos is linked with cols or x-axis of cartesion plane
          c = (int)(y + r * sin(angle));
          d = (int)(x + r * cos(angle));
          //VOTING -------
          if(a>0 && a<grad.rows && b>0 && b<grad.cols){  acc[a][b][r] ++;}
          if(c>0 && c<grad.rows && d>0 && d<grad.cols){  acc[c][d][r] ++;}
        }
      }
    }
  }

  //---------------------------------------------------------------
  // Creating image for 2D-HOUGH Space
  Mat hough2D(grad.rows, grad.cols, CV_32FC1, Scalar(0)) ;

  int vote = 0, radius = 0;
  int max_Vote = 0, max_Y = 0, max_X = 0, max_R = 0;

  //looping to (1)create 2d Hough (2)printing circles above threshold (3)finding strongestCircle centre
  for(int y = 0 ; y < grad.rows ; y++)
  {
    for(int x = 0 ; x < grad.cols ; x++)
    {
			//hough2D.at<float>(y,x) = 50;
      vote=0; radius=0;
      for(int r = rmin ; r < rmax ; r++)
      {
        hough2D.at<float>(y,x) += acc[y][x][r];   //adding the votes to create 2D HOUGH space

        if(acc[y][x][r] > vote){   //Finding the radius with maximum votes in each point of the image
          vote = acc[y][x][r];
          radius = r;
        }
      }

      //We do this to avoid detecting multiple circles at the same point * * *
      if(vote > voteTh){  //Checking if vote is above the threshold
        circle(img_h, Point(x,y), radius, cvScalar(0,255,0), 2);   //Printing the circle
        circle(img_h, Point(x,y), 1, cvScalar(255,0,0), 2);        //Printing centre pt
				int ry = y - radius;
				int rx = x - radius;
				int rht = radius*2;
				int rwt = radius*2;
				circleVec.push_back(Rect(rx,ry,rht,rwt)); //send all circle above threshold

      }

      if(vote > max_Vote){  //Storing the point with the maximim vote in the entire image
        max_Vote = vote;  max_R = radius; max_Y = y;  max_X = x;
      }

    }
  }

  //---------------------------------------------------------------

  //Printing the circle with MAXimum VOTE or the strongestCircle Centre
  cout << endl << "CIRCLES - Maximum vote : " << max_Vote << " at " << max_Y << "," << max_X << "," << max_R << endl;
  circle(img_h, Point(max_X,max_Y), max_R, cvScalar(255,0,255), 2);
	circle(img_h, Point(max_X,max_Y), 2, cvScalar(255,0,255), 2);

	int ry = max_Y - max_R;
	int rx = max_X - max_R;
	int rht = max_R*2;
	int rwt = max_R*2;
	strongestCircle.push_back(Rect(rx,ry,rht,rwt)); //strongest circle

  //normalize(hough2D, hough2D, 0, 255, NORM_MINMAX);

  hough2D.convertTo(hough2D,CV_8UC1);   //Roundind off pixel values
  //std::cout << hough2D << '\n';         //printing hough space pixel values
  //imshow("Hough 2d", hough2D);          //displaying hough space
  imwrite("Hough 2d.jpg", hough2D);     //storing hough space
  hough2D.release();

	img_h.convertTo(img_h,CV_8UC1);
  imshow("hough circle", img_h);
  imwrite("Hough Circles.jpg", img_h);
  return img_h;
}

//*************************************************************************************************************************************************************

Mat threshold(Mat grad)
{
  float pix;
  float pixTh = 70;
  Mat thresh = grad.clone();
  for (int y = 0; y < grad.rows; y++){
    for (int x = 0; x < grad.cols; x++){
      pix = grad.at<float>(y,x);
      thresh.at<float>(y,x) = (pix > pixTh)? 255 : 0;
    }
  }
  //imshow("Threshold", thresh);
	imwrite("Threshold.jpg", thresh);
  return thresh;

}

Mat gradient(Mat image)
{
  // set up tranform kernel

  Mat xKernel = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
  Mat yKernel = (Mat_<float>(3,3) << -1,-2,-1,0,0,0,1,2,1);
  Mat ys = convolution(image, yKernel);
  Mat xs = convolution(image, xKernel);
  Mat grad = xs.clone();
  //looping through to get gradient Magnitude
  float dx,dy;
  for (int y = 0; y < xs.rows; y++){
    for (int x = 0; x < xs.cols; x++){
      dx = xs.at<float>(y,x);
      dy = ys.at<float>(y,x);

      grad.at<float>(y,x) = sqrt((dx*dx)+(dy*dy));
      //cout << "  " << grad.at<float>(y,x) ;
    }//cout<<endl<< "new line"<<endl;
  }
  normalize(grad, grad, 0, 255, NORM_MINMAX);
  return grad;
}

Mat direction (Mat image)
{
  // set up tranform kernel
  Mat xKernel = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
  Mat yKernel = (Mat_<float>(3,3) << -1,-2,-1,0,0,0,1,2,1);
  Mat ys = convolution(image, yKernel);
  Mat xs = convolution(image, xKernel);

  Mat theta = xs.clone();
  float dx,dy, ang;

  for (int y = 0; y < xs.rows; y++){
    for (int x = 0; x < xs.cols; x++){
      dx = xs.at<float>(y,x);
      dy = ys.at<float>(y,x);
      theta.at<float>(y,x) = atan2(dy,dx);// in radians
      //cout << "  " << theta.at<float>(y,x) ;
    }//cout<<endl<< "new line"<<endl;
  }
  return theta;

}

Mat convolution(Mat base_img, Mat kernel){

  Mat new_img = base_img.clone(); //create output image
  new_img.convertTo(new_img, CV_32FC1); //convert to float so we can get larger numbers

  //Loop through the image
  for (int y = 0; y < base_img.rows; y++){
    for (int x = 0; x < base_img.cols; x++){
      float sum = 0;//reset sum at the start of a new pixel
      //Loop through the kernel
      for (int i = -1; i <= 1; i++){
        for (int j = -1; j <= 1; j++){
          //check if you go out of image range
          if (!(y+i < 0 || y+i > base_img.rows || x+j < 0 || x+j > base_img.cols)){
            sum += (base_img.at<uchar>(y+i,x+j) * kernel.at<float>(i+1,j+1));
          }
        }
      }//set the sum to the new image
      new_img.at<float>(y,x) = sum;
    }
  }
  return new_img;
}

int ***malloc3dArray(int dim1, int dim2, int dim3)
{
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
      	for (j = 0; j < dim2; j++) {
        	    array[i][j] = (int *) malloc(dim3 * sizeof(int));
      	}

    }
    return array;
}
