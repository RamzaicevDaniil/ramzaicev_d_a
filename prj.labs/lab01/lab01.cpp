#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <chrono>
int main() {
  int cols = 768;
  int rows = 60;
  cv::Mat img(rows*3, cols, CV_8UC1);
  img = 127;
  std::vector<int> grad(cols);

  int i = 0;
  for (int color = 0; color < 256; color++) {
    grad[i] = color;
    grad[i + 1] = color;
    grad[i + 2] = color;
    i += 3;
  }

  cv::Rect roi1(0, rows, cols, rows);
  cv::Mat imageRoi1(img, roi1);
  cv::Rect roi2(0, rows*2, cols, rows);
  cv::Mat imageRoi2(img, roi2);
  cv::Rect roiMain(0, 0, cols, rows);
  cv::Mat imageRoiMain(img, roiMain);

  // img1
  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::milliseconds ms;
  typedef std::chrono::duration<float> fsec;

  auto t0 = Time::now();
  
  for (int i = 0; i < cols; i++) {
    cv::Point p1(i, 0);
    cv::Point p2(i, rows);
    cv::line(imageRoiMain, p1, p2, cv::Scalar(grad[i]), 1);
  }

  auto t1 = Time::now();
  fsec fs = t1 - t0;
  ms d = std::chrono::duration_cast<ms>(fs);
  std::cout << "gradient" << std::endl;
  std::cout << fs.count() << "s\n";
  std::cout << d.count() << "ms\n";

  // img2

  t0 = Time::now();
  cv::Mat tmp(imageRoi1.size(), CV_32FC1);
  cv::Mat tmp1(imageRoi1.size(), CV_32FC1);
  imageRoiMain.convertTo(tmp, CV_32FC1, 1.0 / 255, 0);
  cv::pow(tmp, 2.2, tmp1);
  cv::Mat tmp2(imageRoi1.size(), CV_8UC1);
  tmp1.convertTo(tmp2, CV_8UC1, 255);
  tmp2.copyTo(imageRoi1);
  cv::imwrite("lab01_tmp.png", tmp2);
  t1 = Time::now();

  fs = t1 - t0;
  d = std::chrono::duration_cast<ms>(fs);
  std::cout << "cv pow" << std::endl;
  std::cout << fs.count() << "s\n";
  std::cout << d.count() << "ms\n";

  // img3

  t0 = Time::now();
  double gamma2 = 2.4;
  double k = 1;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      cv::Point p(i, j);
      imageRoi2.at<uint8_t>(i, j) = cv::saturate_cast<uint8_t>(k * pow(grad[j] * 1.0 / 255.0, gamma2) * 255.0);
    }
  }
  t1 = Time::now();
  fs = t1 - t0;
  d = std::chrono::duration_cast<ms>(fs);
  std::cout << "cmath pow" << std::endl;
  std::cout << fs.count() << "s\n";
  std::cout << d.count() << "ms\n";
  cv::imwrite("lab01.png", img);
}
