// =====================================================================
// gdsd_cli.cpp — CLI: read image (OpenCV) -> run GDSD C++ -> write edge map
// Usage: ./gdsd_cli <in.jpg> <out.npy or out.png> [percentile] [sigma]
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "gdsd_cpp.hpp"

// Write .npy (uint8, shape h x w) — used by the differential test
static bool write_npy(const std::string& path, const std::vector<uint8_t>& data,
                      int h, int w) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    // magic + version
    std::string magic = "\x93NUMPY";
    std::fwrite(magic.data(), 1, 6, fp);
    std::fwrite("\x01\x00", 1, 2, fp);
    char header[256];
    // same format as numpy: "{'descr': '|u1', 'fortran_order': False, 'shape': (h, w), }"
    int n = std::snprintf(header, sizeof(header),
        "{'descr': '|u1', 'fortran_order': False, 'shape': (%d, %d), }",
        h, w);
    // header length must be a multiple of 64 counting: magic(6)+ver(2)+len(2)+header+'\n'
    int header_total = 10 + n + 1;
    int pad = (64 - (header_total % 64)) % 64;
    uint16_t hlen_u16 = (uint16_t)(n + 1 + pad);
    std::fwrite(&hlen_u16, 2, 1, fp);
    std::fwrite(header, 1, n, fp);
    std::fputc('\n', fp);
    for (int i = 0; i < pad; ++i) std::fputc(' ', fp);
    std::fwrite(data.data(), 1, data.size(), fp);
    std::fclose(fp);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <in.jpg> <out.npy> [percentile=90] [sigma=1.4] [grad_thr=0.01]\n", argv[0]);
        return 2;
    }
    double percentile = argc > 3 ? std::atof(argv[3]) : 90.0;
    double sigma = argc > 4 ? std::atof(argv[4]) : 1.4;
    double grad_thr = argc > 5 ? std::atof(argv[5]) : 0.01;

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    int h = img.rows, w = img.cols;

    // uint8 gray -> float [0,1]
    std::vector<double> gray(h * w);
    for (int i = 0; i < h * w; ++i) gray[i] = img.data[i] / 255.0;

    auto res = gdsd::detect_edges(gray, h, w, percentile, sigma, grad_thr, /*radius*/2, /*step*/0.1);

    std::string out = argv[2];
    if (out.size() >= 4 && out.substr(out.size()-4) == ".npy") {
    // write .npy by hand (debug only)
    write_npy(out, res.edge, h, w);
    } else {
    cv::Mat em(h, w, CV_8U, (void*)res.edge.data());
    cv::imwrite(out, em);
    }
    std::printf("done %s: %dx%d edge_px=%zu Tp=%.6f Tlow=%.6f\n",
                argv[1], w, h, (size_t)std::count(res.edge.begin(), res.edge.end(), 1),
                res.Tp, res.Tlow);
    return 0;
}
