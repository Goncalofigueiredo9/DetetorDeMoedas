#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "vc.h"

constexpr double PI = 3.141592653589793;


IVC* mat_to_ivc(const cv::Mat& src) {
    IVC* img = vc_image_new(src.cols, src.rows, 3, 255);
    std::memcpy(img->data, src.data, src.cols * src.rows * 3);
    return img;
}


cv::Mat ivc_to_mat(IVC* img) {
    cv::Mat m(img->height, img->width, CV_8UC3);
    std::memcpy(m.data, img->data, img->width * img->height * 3);
    return m;
}

int main() {

    cv::VideoCapture cap("video1.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Erro a abrir vídeo\n";
        return -1;
    }

    int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = (int)cap.get(cv::CAP_PROP_FPS);
    int delay = 1000 / fps;
    double escala_pixel_mm = 12.875 / 65.0;

    cv::Mat frame;

    while (cap.read(frame) && !frame.empty()) {

        IVC* ivc_bgr = mat_to_ivc(frame);


        IVC* ivc_hsv = vc_image_new(W, H, 3, 255);
        vc_rgb_to_hsv(ivc_bgr, ivc_hsv);
        cv::Mat hsv = ivc_to_mat(ivc_hsv);


        IVC* ivc_gray = vc_image_new(W, H, 1, 255);
        vc_rgb_to_gray(ivc_bgr, ivc_gray);

        cv::Mat gray(H, W, CV_8UC1);
        std::memcpy(gray.data, ivc_gray->data, W * H);


        IVC* mask_red = vc_image_new(W, H, 1, 255);
        IVC* mask_yellow = vc_image_new(W, H, 1, 255);
        IVC* mask_green = vc_image_new(W, H, 1, 255);
        IVC* mask_blue = vc_image_new(W, H, 1, 255);
        IVC* mask_navy = vc_image_new(W, H, 1, 255);
        IVC* mask_black = vc_image_new(W, H, 1, 255);

        vc_hsv_segmentation(ivc_hsv, mask_red, 0, 10, 100, 255, 100, 255);
        vc_hsv_segmentation(ivc_hsv, mask_yellow, 20, 40, 100, 255, 100, 255);
        vc_hsv_segmentation(ivc_hsv, mask_green, 35, 85, 50, 255, 100, 255);
        vc_hsv_segmentation(ivc_hsv, mask_blue, 85, 135, 50, 255, 100, 255);
        vc_hsv_segmentation(ivc_hsv, mask_navy, 90, 130, 20, 120, 60, 255);
        vc_hsv_segmentation(ivc_hsv, mask_black, 0, 255, 0, 255, 0, 50);

        cv::Mat m_red(H, W, CV_8UC1, mask_red->data);
        cv::Mat m_yellow(H, W, CV_8UC1, mask_yellow->data);
        cv::Mat m_green(H, W, CV_8UC1, mask_green->data);
        cv::Mat m_blue(H, W, CV_8UC1, mask_blue->data);
        cv::Mat m_navy(H, W, CV_8UC1, mask_navy->data);
        cv::Mat m_black(H, W, CV_8UC1, mask_black->data);

        gray.setTo(0, m_red);
        gray.setTo(0, m_yellow);
        gray.setTo(0, m_green);
        gray.setTo(0, m_blue);
        gray.setTo(0, m_navy);
        gray.setTo(0, m_black);

        cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2);

        std::vector<cv::Vec3f> circles;
        cv::imshow("Gray para Hough", gray);
        cv::HoughCircles(gray, circles,
            cv::HOUGH_GRADIENT,
            1,
            60,
            150,
            50,
            45,
            100);


        IVC* ivc_rgb = mat_to_ivc(frame);
        struct A { int x, y, r; std::string t; double area, per; };
        std::vector<A> annots;
        int totalCnt = 0;
        double totalVal = 0.0;


        for (auto& c : circles) {
            int cx = cvRound(c[0]);
            int cy = cvRound(c[1]);
            int r_px = cvRound(c[2]);
            double r_mm = r_px * escala_pixel_mm;


            if (r_mm < 11.0 || r_mm > 18.5) continue;

            double area = PI * r_px * r_px;
            double per = 2 * PI * r_px;
            if ((4 * PI * area) / (per * per) < 0.75) continue;

            cv::Mat m_circ = cv::Mat::zeros(frame.size(), CV_8UC1);
            for (int yy = std::max(0, cy - r_px); yy < std::min(H, cy + r_px); ++yy) {
                for (int xx = std::max(0, cx - r_px); xx < std::min(W, cx + r_px); ++xx) {
                    if ((xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= r_px * r_px)
                        m_circ.at<uchar>(yy, xx) = 255;
                }
            }


            cv::Scalar mb = cv::mean(frame, m_circ);
            IVC* pix = vc_image_new(1, 1, 3, 255);
            pix->data[0] = (uchar)mb[0];
            pix->data[1] = (uchar)mb[1];
            pix->data[2] = (uchar)mb[2];
            IVC* pix_hsv = vc_image_new(1, 1, 3, 255);
            vc_rgb_to_hsv(pix, pix_hsv);
            cv::Vec3b hv{ pix_hsv->data[0], pix_hsv->data[1], pix_hsv->data[2] };
            vc_image_free(pix);
            vc_image_free(pix_hsv);

            int h = hv[0], s = hv[1], v = hv[2];

            if ((s > 120 && (h < 5 || h > 170)) || s > 220 || v < 35) continue;


            std::string tipo;
            double val;
            if (r_mm < 11.6) tipo = "1c", val = 0.01;
            else if (r_mm < 12.4) tipo = "2c", val = 0.02;
            else if (r_mm < 13.4) tipo = "5c", val = 0.05;
            else if (r_mm < 14.2) tipo = "10c", val = 0.10;
            else if (r_mm < 15.1) tipo = "20c", val = 0.20;
            else if (r_mm < 15.9) tipo = "50c", val = 0.50;
            else if (r_mm < 16.8) tipo = "1e", val = 1.00;
            else                   tipo = "2e", val = 2.00;

            totalCnt++;
            totalVal += val;
            annots.push_back({ cx,cy,r_px,tipo,area,per });


            vc_draw_circle(ivc_rgb, cx, cy, r_px, 0, 255, 0);
            vc_draw_rectangle(ivc_rgb, cx - r_px, cy - r_px, cx + r_px, cy + r_px, 255, 255, 0);
            vc_draw_circle(ivc_rgb, cx, cy, 2, 255, 0, 255);
        }


        frame = ivc_to_mat(ivc_rgb);
        for (auto& a : annots) {
            std::ostringstream ss;
            ss << a.t << " A=" << std::fixed << std::setprecision(0) << a.area
                << " P=" << std::fixed << std::setprecision(0) << a.per;
            cv::putText(frame, ss.str(),
                cv::Point(a.x - a.r, a.y - a.r - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 2);
            cv::putText(frame, ss.str(),
                cv::Point(a.x - a.r, a.y - a.r - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);
        }
        std::ostringstream tot;
        tot << "Total: " << totalCnt
            << "  Valor: " << std::fixed << std::setprecision(2) << totalVal << " EUR";
        cv::putText(frame, tot.str(), cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 1.0,
            cv::Scalar(0, 0, 0), 3);
        cv::putText(frame, tot.str(), cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 1.0,
            cv::Scalar(255, 255, 255), 1);


        cv::imshow("Moedas Detetadas (Final)", frame);
        if (cv::waitKey(delay) == 'q') break;


        vc_image_free(ivc_bgr);
        vc_image_free(ivc_hsv);
        vc_image_free(ivc_gray);
        vc_image_free(mask_red);
        vc_image_free(mask_yellow);
        vc_image_free(mask_green);
        vc_image_free(mask_blue);
        vc_image_free(mask_navy);
        vc_image_free(mask_black);
        vc_image_free(ivc_rgb);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
