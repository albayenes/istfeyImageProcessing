#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<iostream>
#include "GMModel.h"

using namespace cv;

int main() {
	VideoCapture videoSource("ImageProcessingVehicleCountingUsingOpenCV.mp4");		// declare a VideoCapture object and associate to webcam, 0 => use 1st webcam

	if (!videoSource.isOpened()) {			// check if VideoCapture object was associated to webcam successfully
		std::cout << "error: video was not accessed successfully\n\n";	// if not, print error message to std out
		return(0);														// and exit program
	}

	Mat frameBGR;
	Mat frame;
	//videoSource >> frame; // get a new frame from camera
	//resize(frame, frame, Size(0.0,0.0),0.75,0.75);
	int height = videoSource.get(CV_CAP_PROP_FRAME_HEIGHT);// *0.75;
	int width = videoSource.get(CV_CAP_PROP_FRAME_WIDTH); // *0.75;
	int K = 3;
	GMModel model(K, height, width);
	
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	
	// Filter by Area.
	params.filterByArea = false;
	params.blobColor = 255;
	/*params.minArea = 5;
	params.maxArea = 50000;*/

	//// Filter by Circularity
	//params.filterByCircularity = true;
	//params.minCircularity = 0.1;

	//// Filter by Convexity
	//params.filterByConvexity = true;
	//params.minConvexity = 0.87;

	//// Filter by Inertia
	//params.filterByInertia = true;
	//params.minInertiaRatio = 0.01;

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// SimpleBlobDetector::create creates a smart pointer. 
	// So you need to use arrow ( ->) instead of dot ( . )
	// detector->detect( im, keypoints);

	for (;;)
	{
		videoSource >> frameBGR; // get a new frame from camera
		//resize(frame, frame, Size(0.0, 0.0), 0.75, 0.75);
		cvtColor(frameBGR, frame, CV_BGR2GRAY);
		frame.convertTo(frame, CV_32FC1, 1/255.0);
		blur(frame, frame, Size(3, 3));
		/*for (int i = 0; i < K; i++){
			absdiff(frame, model.meanMatrices[i], model.diffFrame[i]);
			imshow("diffFrame " + std::to_string(i), model.diffFrame[i]);
		}*/

		model.findForeGround(frame);

		

		
		Mat element = getStructuringElement(2, Size(5, 5), Point(0, 0));

		/// Apply the specified morphology operation
		morphologyEx(model.foreGround, model.foreGround, MORPH_CLOSE, element);

		element = getStructuringElement(2, Size(3, 3), Point(0, 0));
		morphologyEx(model.foreGround, model.foreGround, MORPH_OPEN, element);

		//bitwise_xor(model.foreGround, filledForeGround, model.foreGround);
		element = getStructuringElement(2, Size(10, 10), Point(0, 0));

		/// Apply the specified morphology operation
		morphologyEx(model.foreGround, model.foreGround, MORPH_CLOSE, element);

		//imshow("foreGround 1", model.foreGround);

		Mat filledForeGround;
		model.foreGround.copyTo(filledForeGround);
		floodFill(filledForeGround, Point(0, 0), Scalar(255));
		

		//// Invert floodfilled image
		Mat im_floodfill_inv;
		bitwise_not(filledForeGround, im_floodfill_inv);
		
		
		bitwise_or(model.foreGround, im_floodfill_inv, model.foreGround);
//		bitwise_not(model.foreGround, model.foreGround);

		//std::cout << model.foreGround;
		
		imshow("foreGround", model.foreGround);
		
		Mat foreGroundBGR = Mat::zeros(height, width, CV_8UC3);

		std::vector<std::vector<Point> > contours;
		std::vector<Vec4i> hierarchy;

		findContours(model.foreGround, contours, hierarchy,
			CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		/// Approximate contours to polygons + get bounding rects and circles
		std::vector<std::vector<Point> > contours_poly(contours.size());
		std::vector<Rect> boundRect(contours.size());
		std::vector<Point2f> center(contours.size());
		std::vector<float> radius(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
		}


		/// Draw polygonal contour + bonding rects + circles
		Mat drawing = Mat::zeros(model.foreGround.size(), CV_8UC3);
		for (int i = 0; i< contours.size(); i++)
		{
			if (boundRect[i].area() > 3000){
				Scalar color(rand() & 255, rand() & 255, rand() & 255);
				//drawContours(drawing, contours_poly, i, color, 1, 8, std::vector<Vec4i>(), 0, Point());
				rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
				//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
			}
		}

		// iterate through all the top-level contours,
		//// draw each connected component with its own random color
		//if (contours.size() > 0){
		//	int idx = 0;
		//	for (; idx >= 0; idx = hierarchy[idx][0])
		//	{
		//		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		//		drawContours(foreGroundBGR, contours, idx, color, CV_FILLED, 8, hierarchy);
		//	}
		//}
		imshow("foreGroundBGR", drawing);
		imshow("frame", frame);


		//// Set up the detector with default parameters.
	////	SimpleBlobDetector detector;
	//	
	//	//// Detect blobs.
	//	std::vector<KeyPoint> keypoints;
	//	detector->detect(model.foreGround, keypoints);
	//	std::cout << keypoints.size() << std::endl;
	//	//// Draw detected blobs as red circles.
	//	//// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
	//	Mat im_with_keypoints;
	//	drawKeypoints(frameBGR, keypoints, im_with_keypoints, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//	//// Show blobs
	//	imshow("keypoints", im_with_keypoints);

		// weights are summed up
		for (int i = 0; i < K; i++){
			add(model.sumWeights, model.weightMatrices[i], model.sumWeights);
		}

		// weight matrix normalized
		for (int i = 0; i < K; i++){
			divide(model.weightMatrices[i], model.sumWeights, model.weightMatrices[i]);
		}
		
		if (waitKey(3) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}