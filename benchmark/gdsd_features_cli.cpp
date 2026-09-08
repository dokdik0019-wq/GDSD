// =====================================================================
// gdsd_features_cli.cpp — CLI: อ่านภาพ -> GDSD C++ -> เขียน response/gradient/ZC
// Output: <prefix>.response.npy, .d.npy, .e.npy, .zc.npy (float64, h x w)
// ใช้สร้าง soft edge map / PR curve สำหรับ benchmark ODS/OIS/AP
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
// C++ core จาก src/cpp (compile ด้วย -I src/cpp — ดู Makefile build-benchmark)
#include "gdsd_cpp.hpp"

static bool write_f64(const std::string& path, const std::vector<double>& data, int h, int w) {
    // raw little-endian float64 ต่อเนื่อง h*w — อ่านด้วย np.fromfile().reshape(h,w)
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    std::fwrite(data.data(), 1, data.size() * sizeof(double), fp);
    std::fclose(fp);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <in.jpg> <out-prefix> [sigma=1.4] [grad_thr=0.01]\n", argv[0]);
        return 2;
    }
    double sigma    = argc > 3 ? std::atof(argv[3]) : 1.4;
    double grad_thr = argc > 4 ? std::atof(argv[4]) : 0.01;

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    int h = img.rows, w = img.cols;

    std::vector<double> gray(h * w);
    for (int i = 0; i < h * w; ++i) gray[i] = img.data[i] / 255.0;

    gdsd::EdgeResult res = gdsd::detect_edges(gray, h, w, /*percentile*/90.0,
                                              sigma, grad_thr, 2, 0.1);

    // d,e = linear coefficients (gradient) — fit ใหม่เหมือน detect_edges (parity เป๊ะ)
    std::vector<double> blurred;
    gdsd::gaussian_blur(gray, h, w, sigma, blurred);
    int radius = 2, win = 2*radius+1;
    auto pinv = gdsd::design_pinv(radius, 0.1);
    std::vector<double> pad((h+2*radius)*(w+2*radius));
    auto pat = [&](int y, int x) -> double {
        return blurred[std::min(std::max(y,0),h-1)*w + std::min(std::max(x,0),w-1)];
    };
    for (int y = -radius; y < h+radius; ++y)
        for (int x = -radius; x < w+radius; ++x)
            pad[(y+radius)*(w+2*radius)+(x+radius)] = pat(y,x);
    std::vector<double> d(h*w), e(h*w);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double patch[25]; int idx=0;
            for (int dy=0; dy<win; ++dy)
                for (int dx=0; dx<win; ++dx)
                    patch[idx++] = pad[(y+dy)*(w+2*radius)+(x+dx)];
            double coef[6]={0,0,0,0,0,0};
            for (int i=0;i<6;++i)
                for (int n=0;n<win*win;++n) coef[i]+=pinv[i][n]*patch[n];
            int p=y*w+x; d[p]=coef[3]; e[p]=coef[4];
        }

    // ZC map (quantized gradient direction + sign change เหมือน detect_edges)
    std::vector<double> nb1(h*w, 0.0), nb2(h*w, 0.0);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int p = y*w + x;
            double g = std::hypot(d[p], e[p]);
            if (g <= grad_thr) continue;
            double dx_ = d[p], dy_ = e[p];
            double mag = std::hypot(dx_, dy_);
            if (mag <= 1e-12) continue;
            int step_y = (int)std::nearbyint(dy_ / mag);
            int step_x = (int)std::nearbyint(dx_ / mag);
            if (step_y == 0 && step_x == 0) continue;
            int y1 = std::min(std::max(y + step_y, 0), h-1);
            int x1 = std::min(std::max(x + step_x, 0), w-1);
            int y2 = std::min(std::max(y - step_y, 0), h-1);
            int x2 = std::min(std::max(x - step_x, 0), w-1);
            nb1[p] = res.response[y1*w + x1];
            nb2[p] = res.response[y2*w + x2];
        }
    std::vector<uint8_t> zc(h*w, 0);
    for (int p = 0; p < h*w; ++p)
        if (nb1[p] * nb2[p] < 0.0) zc[p] = 1;

    std::string pre = argv[2];
    write_f64(pre + ".response.f64", res.response, h, w);
    write_f64(pre + ".d.f64", d, h, w);
    write_f64(pre + ".e.f64", e, h, w);
    write_f64(pre + ".zc.f64", std::vector<double>(zc.begin(), zc.end()), h, w);
    std::printf("done %s: %dx%d edge_px=%zu zc_px=%zu\n", argv[1], w, h,
                (size_t)std::count(res.edge.begin(), res.edge.end(), 1),
                (size_t)std::count(zc.begin(), zc.end(), 1));
    return 0;
}
