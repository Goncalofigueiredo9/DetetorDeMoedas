#pragma once

#include <opencv2/opencv.hpp>

typedef struct {
    unsigned char* data;
    int width, height;
    int channels;
    int levels;
    int bytesperline;
} IVC;


IVC* vc_image_new(int width, int height, int channels, int levels);
IVC* vc_image_free(IVC* image);


int vc_rgb_to_hsv(IVC* src, IVC* dst);


int vc_rgb_to_gray(IVC* src, IVC* dst);


int vc_hsv_segmentation(IVC* src, IVC* dst,
    int hmin, int hmax,
    int smin, int smax,
    int vmin, int vmax);


int vc_draw_circle(IVC* image, int cx, int cy, int r, int R, int G, int B);
int vc_draw_rectangle(IVC* image, int x1, int y1, int x2, int y2, int R, int G, int B);