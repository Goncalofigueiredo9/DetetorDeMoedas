#include "vc.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <opencv2/opencv.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


IVC* vc_image_new(int width, int height, int channels, int levels) {

    IVC* image = (IVC*)malloc(sizeof(IVC));
    if (!image || levels <= 0 || levels > 255) return NULL;


    image->width = width;
    image->height = height;
    image->channels = channels;
    image->levels = levels;
    image->bytesperline = width * channels;


    image->data = (unsigned char*)malloc(width * height * channels);
    if (!image->data)

        return vc_image_free(image);
    return image;
}


IVC* vc_image_free(IVC* image) {
    if (image) {
        free(image->data);
        free(image);
    }
    return NULL;
}



int vc_rgb_to_hsv(IVC* src, IVC* dst) {

    if (!src || !dst ||
        src->channels != 3 || dst->channels != 3 ||
        src->width != dst->width ||
        src->height != dst->height)
        return 0;


    for (int y = 0; y < src->height; y++) {
        for (int x = 0; x < src->width; x++) {
            int pos = y * src->bytesperline + x * 3;

            float r = src->data[pos + 2] / 255.0f;
            float g = src->data[pos + 1] / 255.0f;
            float b = src->data[pos + 0] / 255.0f;


            float mx = std::max({ r,g,b });
            float mn = std::min({ r,g,b });
            float delta = mx - mn;

            float h = 0.0f, s = 0.0f, v = mx;

            if (delta > 1e-6f) {
                if (mx == r) h = 60.0f * fmodf((g - b) / delta, 6.0f);
                else if (mx == g) h = 60.0f * (((b - r) / delta) + 2.0f);
                else              h = 60.0f * (((r - g) / delta) + 4.0f);
                if (h < 0.0f) h += 360.0f;
                s = delta / mx;
            }

            dst->data[pos + 0] = (unsigned char)(h / 360.0f * 255.0f);
            dst->data[pos + 1] = (unsigned char)(s * 255.0f);
            dst->data[pos + 2] = (unsigned char)(v * 255.0f);
        }
    }
    return 1;
}



int vc_rgb_to_gray(IVC* src, IVC* dst) {
    if (!src || !dst ||
        src->channels != 3 || dst->channels != 1 ||
        src->width != dst->width ||
        src->height != dst->height)
        return 0;


    for (int y = 0; y < src->height; y++) {
        for (int x = 0; x < src->width; x++) {
            int pos = y * src->bytesperline + x * 3;
            unsigned char b = src->data[pos + 0];
            unsigned char g = src->data[pos + 1];
            unsigned char r = src->data[pos + 2];
            unsigned char gray = static_cast<unsigned char>(
                0.114f * b + 0.587f * g + 0.299f * r
                );
            dst->data[y * dst->bytesperline + x] = gray;
        }
    }
    return 1;
}


int vc_hsv_segmentation(IVC* src, IVC* dst,
    int hmin, int hmax,
    int smin, int smax,
    int vmin, int vmax) {
    if (!src || !dst ||
        src->channels != 3 || dst->channels != 1 ||
        src->width != dst->width ||
        src->height != dst->height)
        return 0;

    for (int y = 0; y < src->height; y++) {
        for (int x = 0; x < src->width; x++) {
            int ps = y * src->bytesperline + x * 3;
            unsigned char h = src->data[ps + 0];
            unsigned char s = src->data[ps + 1];
            unsigned char v = src->data[ps + 2];


            dst->data[y * dst->bytesperline + x] =
                (h >= hmin && h <= hmax &&
                    s >= smin && s <= smax &&
                    v >= vmin && v <= vmax) ? 255 : 0;
        }
    }
    return 1;
}



int vc_draw_circle(IVC* image, int cx, int cy, int r,
    int R, int G, int B) {
    for (double t = 0; t < 2 * M_PI; t += 0.01) {
        int x = cx + (int)(r * cos(t));
        int y = cy + (int)(r * sin(t));
        if (x >= 0 && x < image->width &&
            y >= 0 && y < image->height) {
            int pos = y * image->bytesperline + x * 3;
            image->data[pos + 0] = (unsigned char)B;
            image->data[pos + 1] = (unsigned char)G;
            image->data[pos + 2] = (unsigned char)R;
        }
    }
    return 1;
}



int vc_draw_rectangle(IVC* image,
    int x1, int y1,
    int x2, int y2,
    int R, int G, int B) {

    for (int x = x1; x <= x2; x++) {
        if (y1 >= 0 && y1 < image->height && x >= 0 && x < image->width) {
            int i = (y1 * image->width + x) * 3;
            image->data[i + 0] = (unsigned char)B;
            image->data[i + 1] = (unsigned char)G;
            image->data[i + 2] = (unsigned char)R;
        }
        if (y2 >= 0 && y2 < image->height && x >= 0 && x < image->width) {
            int i = (y2 * image->width + x) * 3;
            image->data[i + 0] = (unsigned char)B;
            image->data[i + 1] = (unsigned char)G;
            image->data[i + 2] = (unsigned char)R;
        }
    }

    for (int y = y1; y <= y2; y++) {
        if (x1 >= 0 && x1 < image->width && y >= 0 && y < image->height) {
            int i = (y * image->width + x1) * 3;
            image->data[i + 0] = (unsigned char)B;
            image->data[i + 1] = (unsigned char)G;
            image->data[i + 2] = (unsigned char)R;
        }
        if (x2 >= 0 && x2 < image->width && y >= 0 && y < image->height) {
            int i = (y * image->width + x2) * 3;
            image->data[i + 0] = (unsigned char)B;
            image->data[i + 1] = (unsigned char)G;
            image->data[i + 2] = (unsigned char)R;
        }
    }
    return 1;
}