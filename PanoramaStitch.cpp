// PanoramaStitch.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;


void stitch_image(Mat image1, Mat image2) {

    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    std::vector<cv::KeyPoint> lastFramekeypoints1, lastFramekeypoints2;
    cv::Mat lastFrameDescriptors1, lastFrameDescriptors2;
    std::vector<cv::DMatch > matches, resultmatches;
    int movementDirection = 0;

    // finds keypoints and their disriptorss
    detector->detectAndCompute(image1, noArray(), lastFramekeypoints1, lastFrameDescriptors1);
    detector->detectAndCompute(image2, noArray(), lastFramekeypoints2, lastFrameDescriptors2);

    // match the descriptor between two images
    matcher->match(lastFrameDescriptors1, lastFrameDescriptors2, matches);

    std::vector<cv::Point2d> good_point1, good_point2;
    good_point1.reserve(matches.size());
    good_point2.reserve(matches.size());

    //calculation of max and min distances between keypoints
    double max_dist = 0; double min_dist = 100;
    for (const auto& m : matches)
    {
        double dist = m.distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    // filter out good points
    // distance which is less than or equals the min_dist*1.5
    // if this value is increased more keypoits are detected.
    for (const auto& m : matches)
    {
        if (m.distance <= 1.5 * min_dist)
        {
            good_point1.push_back(lastFramekeypoints1.at(m.queryIdx).pt);
            good_point2.push_back(lastFramekeypoints2.at(m.trainIdx).pt);
        }
    }

    // crop rectangle constructor.
    cv::Rect croppImg1(0, 0, image1.cols, image1.rows);
    cv::Rect croppImg2(0, 0, image2.cols, image2.rows);

    int imgWidth = image1.cols;
    for (int i = 0; i < good_point1.size(); ++i)
    {
        if (good_point1[i].x < imgWidth)
        {
            croppImg1.width = good_point1.at(i).x;
            croppImg2.x = good_point2[i].x;
            croppImg2.width = image2.cols - croppImg2.x;
            movementDirection = good_point1[i].y - good_point2[i].y;
            imgWidth = good_point1[i].x;
        }
    }
    image1 = image1(croppImg1);
    image2 = image2(croppImg2);
    int maxHeight = image1.rows > image2.rows ? image1.rows : image2.rows;
    int maxWidth = image1.cols + image2.cols;
    cv::Mat result = cv::Mat::zeros(cv::Size(maxWidth, maxHeight + abs(movementDirection)), CV_8UC3);
    if (movementDirection > 0)
    {
        cv::Mat half1(result, cv::Rect(0, 0, image1.cols, image1.rows));
        image1.copyTo(half1);
        cv::Mat half2(result, cv::Rect(image1.cols, abs(movementDirection), image2.cols, image2.rows));
        image2.copyTo(half2);
    }
    else
    {
        cv::Mat half1(result, cv::Rect(0, abs(movementDirection), image1.cols, image1.rows));
        image1.copyTo(half1);
        cv::Mat half2(result, cv::Rect(image1.cols, 0, image2.cols, image2.rows));
        image2.copyTo(half2);
    }
    imshow("Stitched Image", result);
}

int main()
{
    VideoCapture cap1("resources/sample1-B.mp4"); //Path to frame1
    VideoCapture cap2("resources/sample1-A.mp4"); //Path to frame2
    int fps = cap1.get(5);
    //VideoWriter video("out.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(126, 116));//Declaring an object of VideoWriter class//
    Mat image1, image2, final_image;
    while (1) {
        cap1 >> image1;
        if (image1.empty()) break;
        cap2 >> image2;
        if (image2.empty()) break;
        //imshow("Image1", image1);
        //imshow("Image2", image2);
        stitch_image(image1, image2);
        int k = waitKey(0);
        if (k == 'e'){
            return 0;
        }
    }
    
    return 0;
}