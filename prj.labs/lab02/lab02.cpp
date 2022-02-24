#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
int main() {
  const std::string pathToDataFolder = "../../../data/";
  const std::string fileName1 = pathToDataFolder + "cross_0256x0256.png";
  const std::string fileName2 = "cross_0256x0256_025.jpg";
  const std::string fileName3 = "cross_0256x0256_png_channels.png";
  const std::string fileName4 = "cross_0256x0256_jpg_channels.png";
  const std::string fileName5 = "cross_0256x0256_hists.png";
  cv::Mat img = cv::imread(fileName1);
  cv::imwrite(fileName2, img, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 25});

  cv::Mat empty_img(512, 512, CV_8UC3);
  std::vector<cv::Mat> bgr_planes;
  cv::Rect roilu(0, 0, 256, 256);
  cv::Mat imglu(empty_img, roilu);
  cv::Rect roill(256, 0, 256, 256);
  cv::Mat imgll(empty_img, roill);
  cv::Rect roiru(0, 256, 256, 256);
  cv::Mat imgru(empty_img, roiru);
  cv::Rect roirl(256, 256, 256, 256);
  cv::Mat imgrl(empty_img, roirl);

  std::vector<cv::Mat> b_channel, g_channel, r_channel;
  cv::split(img, b_channel);
  cv::split(img, g_channel);
  cv::split(img, r_channel);

  b_channel[1] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  b_channel[2] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  cv::merge(b_channel, imgrl);
  g_channel[0] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  g_channel[2] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  cv::merge(g_channel, imgll);
  r_channel[0] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  r_channel[1] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  cv::merge(r_channel, imgru);

  img.copyTo(imglu);
  cv::imwrite(fileName3, empty_img);

  // jpg image
  img = cv::imread(fileName2);

  cv::split(img, b_channel);
  cv::split(img, g_channel);
  cv::split(img, r_channel);

  b_channel[1] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1); 
  b_channel[2] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  cv::merge(b_channel, imgrl);
  g_channel[0] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  g_channel[2] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  cv::merge(g_channel, imgll);
  r_channel[0] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  r_channel[1] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  cv::merge(r_channel, imgru);
  img.copyTo(imglu);

  cv::imwrite(fileName4, empty_img);

  // hist
  cv::Mat imgPng = cv::imread(fileName1);
  cv::Mat imgJpg = cv::imread(fileName2);
  std::vector<cv::Mat> channelsPng;
  cv::split(imgPng, channelsPng);
  std::vector<cv::Mat> channelsJpg;
  cv::split(imgJpg, channelsJpg);
  std::vector<std::vector<int>> histJpg(3, std::vector<int>(256, 0));
  std::vector<std::vector<int>> histPng(3, std::vector<int>(256, 0));
  for (int channel = 0; channel < 3; channel++) {
    for (int r = 0; r < imgJpg.rows; r++) {
      for (int c = 0; c < imgJpg.cols; c++) {
        int valueJpg = channelsJpg[channel].at<uint8_t>(r, c);
        histJpg[channel][valueJpg]++;
        int valuePng = channelsPng[channel].at<uint8_t>(r, c);
        histPng[channel][valuePng]++;
      }
    }
  }

  //hist graph
  cv::Mat histGraph(513, 256, CV_8UC3);
  histGraph = cv::Scalar(0, 0, 0);
  //cv::Point p1(0, 257);
  //cv::Point p2(259, 259);
  //cv::line(histGraph, p1, p2, cv::Scalar(255, 255, 255), 1);

  cv::Rect histPngRect(0, 0, 256, 256);
  cv::Mat histPngRoi(histGraph, histPngRect);
  cv::Rect histJpgRect(0, 257, 256, 256);
  cv::Mat histJpgRoi(histGraph, histJpgRect);

  int maxValueJpg = 0;
  int maxValuePng = 0;
  for (int channel = 0; channel < 3; channel++) {
    for (int value = 0; value < 256; value++) {
      if (maxValuePng < histPng[channel][value])
        maxValuePng = histPng[channel][value];
      if (maxValueJpg < histJpg[channel][value])
        maxValueJpg = histJpg[channel][value];
    }
  }

  int new_max = 228;
  for (int channel = 0; channel < 3; channel++) {
    for (int value = 0; value < 256; value++) {
      histPng[channel][value] = std::min(std::max(256 - histPng[channel][value] * new_max / maxValuePng, 0), 255);
      histJpg[channel][value] = std::min(std::max(256 - histJpg[channel][value] * new_max / maxValueJpg, 0), 255);
      //std::cout << histPng[channel][value] << " ";
    }
  }
  

  std::vector<cv::Scalar> channelColors;
  channelColors.push_back(cv::Scalar(255, 0, 0));
  channelColors.push_back(cv::Scalar(0, 255, 0));
  channelColors.push_back(cv::Scalar(0, 0, 255));
  for (int channel = 0; channel < 3; channel++) {
    cv::Point p1(0, histPng[channel][0]);
    for (int value = 0; value < 256; value++) {
      histPngRoi.at<cv::Vec3b>(p1)[channel] = 255;
      cv::Point p2(value, histPng[channel][value]);
      cv::line(histPngRoi, p1, p2, channelColors[channel], 1);
      p1 = p2;
      std::cout << channel << " " << value << std::endl;
    }
  }
  for (int channel = 0; channel < 3; channel++) {
    cv::Point p1(0, histJpg[channel][0]);
    for (int value = 0; value < 256; value++) {
      histJpgRoi.at<cv::Vec3b>(p1)[channel] = 255;
      cv::Point p2(value, histJpg[channel][value]);
      cv::line(histJpgRoi, p1, p2, channelColors[channel], 1);
      p1 = p2;
      std::cout << channel << " " << value << std::endl;
    }
  }

  cv::imwrite(fileName5, histGraph);
}