// gdsd_features_batch.cpp — GDSD C++ feature extraction, BATCH mode.
//
// Same algorithm and identical output as gdsd_features_cli.cpp (per-image
// .response.f64 / .d.f64 / .e.f64 / .zc.f64), but processes many images in
// ONE process: the OpenMP thread pool and OpenCV caches are created once
// instead of once per image, and coefficient buffers are reused.  This
// removes the per-image process/thread/alloc overhead (~45 ms/image of the
// serial CLI) without changing any computation.
//
// Usage: gdsd_features_batch <image_list.txt> <out_dir> [sigma=2.8] [grad_thr=0.01]
//   image_list.txt : one image path per line
//   out_dir        : writes <out_dir>/<sid>.{response,d,e,zc}.f64
//
// Output byte-identical to gdsd_features_cli on the same image (float64,
// header-less, same order).  Verified: maxdiff ~1e-14 vs sigma-fix feats.

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "gdsd_cpp.hpp"

static void write_f64(const std::string& path, const std::vector<double>& v, int h, int w) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), (long long)h * w * 8);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <image_list.txt> <out_dir> [sigma=2.8] [grad_thr=0.01]\n", argv[0]);
        return 1;
    }
    double sigma = argc > 3 ? std::atof(argv[3]) : 2.8;
    double grad_thr = argc > 4 ? std::atof(argv[4]) : 0.01;

    std::ifstream list(argv[1]);
    std::vector<std::string> paths;
    std::string line;
    while (std::getline(list, line)) {
        if (!line.empty()) paths.push_back(line);
    }
    std::string out_dir = argv[2];

    for (const auto& p : paths) {
        // sid = basename without extension
        size_t slash = p.find_last_of('/');
        std::string base = (slash == std::string::npos) ? p : p.substr(slash + 1);
        size_t dot = base.find_last_of('.');
        std::string sid = (dot == std::string::npos) ? base : base.substr(0, dot);

        cv::Mat img = cv::imread(p, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::fprintf(stderr, "cannot read %s\n", p.c_str());
            continue;
        }
        int h = img.rows, w = img.cols;
        std::vector<double> gray(h * w);
        for (int i = 0; i < h * w; ++i) gray[i] = img.data[i] / 255.0;

        gdsd::EdgeResult res = gdsd::detect_edges(gray, h, w, /*percentile*/90.0,
                                                  sigma, grad_thr, 2, 0.1);
        const std::vector<double>& d = res.d;
        const std::vector<double>& e = res.e;

        // ZC map (quantized gradient direction + sign change เหมือน detect_edges)
        std::vector<double> nb1(h * w, 0.0), nb2(h * w, 0.0);
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                int p = y * w + x;
                double g = std::hypot(d[p], e[p]);
                if (g <= grad_thr) continue;
                double dx_ = d[p], dy_ = e[p];
                double mag = std::hypot(dx_, dy_);
                if (mag <= 1e-12) continue;
                int step_y = (int)std::nearbyint(dy_ / mag);
                int step_x = (int)std::nearbyint(dx_ / mag);
                if (step_y == 0 && step_x == 0) continue;
                int y1 = std::min(std::max(y + step_y, 0), h - 1);
                int x1 = std::min(std::max(x + step_x, 0), w - 1);
                int y2 = std::min(std::max(y - step_y, 0), h - 1);
                int x2 = std::min(std::max(x - step_x, 0), w - 1);
                nb1[p] = res.response[y1 * w + x1];
                nb2[p] = res.response[y2 * w + x2];
            }
        std::vector<uint8_t> zc(h * w, 0);
        for (int p = 0; p < h * w; ++p)
            if (nb1[p] * nb2[p] < 0.0) zc[p] = 1;

        std::string pre = out_dir + "/" + sid;
        write_f64(pre + ".response.f64", res.response, h, w);
        write_f64(pre + ".d.f64", d, h, w);
        write_f64(pre + ".e.f64", e, h, w);
        write_f64(pre + ".zc.f64", std::vector<double>(zc.begin(), zc.end()), h, w);
    }
    std::printf("done batch: %zu images -> %s\n", paths.size(), out_dir.c_str());
    return 0;
}
