#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <string>
#include "single_include/nlohmann/json.hpp"
#include <math.h> 
#include <algorithm>

int main() {
  int cols = 150;
  int rows = 150;
  std::vector<cv::Rect> rects;
  std::vector<cv::Mat> rois(6);
  cv::Mat img(rows * 2, cols * 3, CV_8UC1);
  for (int i = 0; i < 6; i++) {
    int j_ = i / 3;
    int i_ = i % 3;
    cv::Rect roi(i_ * 150, j_ * 150, 150, 150);
    rois[i] = cv::Mat(img, roi);
  }

  
  rois[0] = 0;
  rois[4] = 0;
  rois[1] = 127;
  rois[5] = 127;
  rois[2] = 255;
  rois[3] = 255;

  cv::circle(rois[1], cv::Point(75, 75), 70, cv::Scalar(0), -1);
  cv::circle(rois[2], cv::Point(75, 75), 70, cv::Scalar(127), -1);
  cv::circle(rois[5], cv::Point(75, 75), 70, cv::Scalar(0), -1);
  cv::circle(rois[3], cv::Point(75, 75), 70, cv::Scalar(127), -1);
  cv::circle(rois[4], cv::Point(75, 75), 70, cv::Scalar(255), -1);
  cv::circle(rois[0], cv::Point(75, 75), 70, cv::Scalar(255), -1);

  cv::imwrite("I.png", img);

  cv::Mat img1tmp1;
  cv::Mat img1;
  cv::Mat img1tmp2;
  cv::Mat img1tmp3;
  img.copyTo(img1tmp1);
  img1tmp1.convertTo(img1tmp2, CV_32F);
  int kernelSize = 3;
  cv::Mat kernel(kernelSize, kernelSize, CV_32F);
  kernel = 0;
  kernel.at<float>(0, 0) = 1.0;
  kernel.at<float>(0, 1) = -2.0;
  kernel.at<float>(0, 2) = 1.0;
  kernel.at<float>(2, 0) = 1.0;
  kernel.at<float>(2, 1) = -2.0;
  kernel.at<float>(2, 2) = 1.0;
  cv::Point anchor(-1, -1);
  int delta = 0;
  filter2D(img1tmp2, img1tmp3, -1, kernel, anchor, delta, cv::BORDER_DEFAULT);
  cv::normalize(img1tmp3, img1, 0, 255, cv::NORM_MINMAX, CV_32F);

  cv::imwrite("I1.png", img1);
  kernel = 0;
  kernel.at<float>(0, 0) = 1.0;
  kernel.at<float>(1, 0) = -2.0;
  kernel.at<float>(2, 0) = 1.0;
  kernel.at<float>(0, 2) = 1.0;
  kernel.at<float>(1, 2) = 2.0;
  kernel.at<float>(2, 2) = 1.0;
  cv::Mat img2tmp;
  cv::Mat img2;
  filter2D(img1tmp2, img2tmp, -1, kernel, anchor, delta, cv::BORDER_DEFAULT);
  cv::normalize(img2tmp, img2, 0, 255, cv::NORM_MINMAX, CV_32F);

  cv::imwrite("I2.png", img2);

  //resMask.size(), CV_8UC1, cv::Scalar(0)

  cv::Mat img3tmp(img1.size(), CV_32F, cv::Scalar(0));
  cv::Mat img3;
  for (int r = 0; r < img1.rows; ++r) {
    for (int c = 0; c < img1.cols; ++c) {
      img3tmp.at<float>(r, c) = std::pow(std::pow(img1.at<float>(r, c), 2) + std::pow(img1.at<float>(r, c), 2), 0.5);  
    }
  }

  cv::normalize(img3tmp, img3, 0, 255, cv::NORM_MINMAX, CV_32F);
  cv::imwrite("I3.png", img3);

}