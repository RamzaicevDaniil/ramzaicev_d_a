#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
int main() {
  const std::string pathToDataFolder = "../../../data/";
  const std::string fileName1 = pathToDataFolder + "cross_0256x0256.png";
  const std::string fileName2 = "lab03_rgb.png";
  const std::string fileName3 = "lab03_gre.png";
  const std::string fileName4 = "lab03_gre_res.png";
  const std::string fileName5 = "lab03_rgb_res.png";
  const std::string fileName6 = "lab03_viz_func.png";

  cv::Mat imgRGB = cv::imread(fileName1);
  cv::imwrite(fileName2, imgRGB);

  cv::Mat imgGray;
  cv::cvtColor(imgRGB, imgGray, cv::COLOR_BGR2GRAY);
  cv::imwrite(fileName3, imgGray);

  std::vector<uint8_t> lut(256, 0);

  for (int i = 0; i < 256; i++) {
    double d = 100.0;
    double m = 255.0;
    lut[i] = cv::saturate_cast<uint8_t>(255.0 * exp(-pow(i - m, 2)/(d*d)));
    //std::cout << int(lut[i]) << " ";
  }

  cv::Mat imgRGBCor;
  cv::LUT(imgRGB, lut, imgRGBCor);
  cv::Mat imgGrayCor;
  cv::LUT(imgGray, lut, imgGrayCor);

  cv::imwrite(fileName4, imgGrayCor);
  cv::imwrite(fileName5, imgRGBCor);

  cv::Mat graph256x256(256, 256, CV_8UC1);
  graph256x256 = 0;
  for (int i = 0; i < 256; i++) {
    cv::Point p1(i, 255 - int(lut[i]));
    graph256x256.at<uint8_t>(p1) = 255;
  }

  cv::Mat graph512x512(512, 512, CV_8UC1);
  cv::resize(graph256x256, graph512x512, cv::Size(512, 512), cv::INTER_NEAREST);

  cv::imwrite(fileName6, graph512x512);
}