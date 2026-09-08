// =====================================================================
// gdsd_features_wls.cpp — GDSD v3 feature extractor (Gaussian-weighted LSQ)
//
// v3 keeps the full GDSD v2 pipeline (blur -> quadratic fit -> response
// R = 2ad^2+2cde+2be^2 -> zero crossing -> gm@ZC soft map) but replaces
// the UNWEIGHTED least-squares fit of the local window with a
// GAUSSIAN-WEIGHTED least-squares fit:
//
//     w(r,c) = exp(-(r^2+c^2) / (2 sigma_w^2))
//     pinv_wls = (A^T W A)^{-1} A^T W
//
// so pixels near the window centre dominate the surface estimate.  This
// is the CHEV/facet-style weighting (same idea as the earlier WLS spike).
//
// Output: <prefix>.response.f64, .d.f64, .e.f64, .zc.f64 (float64, h x w)
//   response = R from the WLS fit; d,e = WLS gradient; zc = sign-change
//   zero crossing along the quantized gradient direction (same rule as v1/v2).
//
// Usage:
//   gdsd_features_wls <in.jpg> <out-prefix> [sigma=1.4] [grad_thr=0.01]
//                     [sigma_w=1.4]
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
// C++ core from src/cpp (blur + design matrix helpers); compile with -I src/cpp
#include "gdsd_cpp.hpp"

static bool write_f64(const std::string& path, const std::vector<double>& data, int h, int w) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    std::fwrite(data.data(), 1, data.size() * sizeof(double), fp);
    std::fclose(fp);
    return true;
}

// Gaussian weights over the window offsets (r, c).
static std::vector<double> gaussian_weights(int radius, double sigma_w) {
    int win = 2 * radius + 1;
    std::vector<double> w(win * win);
    double denom = 2.0 * sigma_w * sigma_w;
    int idx = 0;
    for (int r = -radius; r <= radius; ++r)
        for (int c = -radius; c <= radius; ++c)
            w[idx++] = std::exp(-(double)(r * r + c * c) / denom);
    return w;
}

// Weighted pseudo-inverse (A^T W A)^{-1} A^T W for the quadratic design A.
static std::vector<std::vector<double>> design_pinv_wls(int radius, double step,
                                                        const std::vector<double>& w) {
    int win = 2 * radius + 1;
    int N = win * win;
    std::vector<std::vector<double>> A(N, std::vector<double>(6));
    int idx = 0;
    for (int r = -radius; r <= radius; ++r)
        for (int c = -radius; c <= radius; ++c) {
            double x = c * step, y = r * step;
            A[idx][0] = x * x; A[idx][1] = y * y; A[idx][2] = x * y;
            A[idx][3] = x;     A[idx][4] = y;     A[idx][5] = 1.0;
            ++idx;
        }
    // AtW (6 x N) = A^T W
    std::vector<std::vector<double>> AtW(6, std::vector<double>(N, 0.0));
    for (int i = 0; i < 6; ++i)
        for (int n = 0; n < N; ++n)
            AtW[i][n] = A[n][i] * w[n];
    // AtWA (6 x 6) = A^T W A
    std::vector<std::vector<double>> AtWA(6, std::vector<double>(6, 0.0));
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            for (int n = 0; n < N; ++n)
                AtWA[i][j] += AtW[i][n] * A[n][j];
    // invert AtWA (Gauss-Jordan)
    std::vector<std::vector<double>> inv(6, std::vector<double>(6, 0.0));
    for (int i = 0; i < 6; ++i) inv[i][i] = 1.0;
    for (int col = 0; col < 6; ++col) {
        int piv = col;
        for (int r = col + 1; r < 6; ++r)
            if (std::fabs(AtWA[r][col]) > std::fabs(AtWA[piv][col])) piv = r;
        std::swap(AtWA[col], AtWA[piv]); std::swap(inv[col], inv[piv]);
        double dd = AtWA[col][col];
        for (int j = 0; j < 6; ++j) { AtWA[col][j] /= dd; inv[col][j] /= dd; }
        for (int r = 0; r < 6; ++r) {
            if (r == col) continue;
            double f = AtWA[r][col];
            for (int j = 0; j < 6; ++j) { AtWA[r][j] -= f * AtWA[col][j]; inv[r][j] -= f * inv[col][j]; }
        }
    }
    // pinv_wls = inv(AtWA) * AtW
    std::vector<std::vector<double>> pinv(6, std::vector<double>(N, 0.0));
    for (int i = 0; i < 6; ++i)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < 6; ++k) pinv[i][n] += inv[i][k] * AtW[k][n];
    return pinv;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <in.jpg> <out-prefix> [sigma=1.4] [grad_thr=0.01] [sigma_w=1.4]\n",
            argv[0]);
        return 2;
    }
    double sigma    = argc > 3 ? std::atof(argv[3]) : 1.4;
    double grad_thr = argc > 4 ? std::atof(argv[4]) : 0.01;
    double sigma_w  = argc > 5 ? std::atof(argv[5]) : 1.4;

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    int h = img.rows, w = img.cols;

    std::vector<double> gray(h * w);
    for (int i = 0; i < h * w; ++i) gray[i] = img.data[i] / 255.0;

    // Stage 0/1: blur + WLS quadratic fit (single pass, coefficients a..f)
    std::vector<double> blurred;
    gdsd::gaussian_blur_5x5(gray, h, w, sigma, blurred);
    int radius = 2, win = 2 * radius + 1;
    auto weights = gaussian_weights(radius, sigma_w);
    auto pinv = design_pinv_wls(radius, 0.1, weights);

    std::vector<double> pad((h + 2 * radius) * (w + 2 * radius));
    auto pat = [&](int y, int x) -> double {
        return blurred[std::min(std::max(y, 0), h - 1) * w + std::min(std::max(x, 0), w - 1)];
    };
    for (int y = -radius; y < h + radius; ++y)
        for (int x = -radius; x < w + radius; ++x)
            pad[(y + radius) * (w + 2 * radius) + (x + radius)] = pat(y, x);

    std::vector<double> a(h * w), b(h * w), c(h * w), d(h * w), e(h * w), f(h * w);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double patch[25]; int idx = 0;
            for (int dy = 0; dy < win; ++dy)
                for (int dx = 0; dx < win; ++dx)
                    patch[idx++] = pad[(y + dy) * (w + 2 * radius) + (x + dx)];
            double coef[6] = {0, 0, 0, 0, 0, 0};
            for (int i = 0; i < 6; ++i)
                for (int n = 0; n < win * win; ++n) coef[i] += pinv[i][n] * patch[n];
            int p = y * w + x;
            a[p] = coef[0]; b[p] = coef[1]; c[p] = coef[2];
            d[p] = coef[3]; e[p] = coef[4]; f[p] = coef[5];
        }

    // Response R = grad^T H grad from the WLS coefficients.
    std::vector<double> response(h * w);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < h * w; ++p)
        response[p] = 2.0 * a[p] * d[p] * d[p]
                    + 2.0 * c[p] * d[p] * e[p]
                    + 2.0 * b[p] * e[p] * e[p];

    // ZC map (quantized gradient direction + sign change, same rule as v1/v2).
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
            nb1[p] = response[y1 * w + x1];
            nb2[p] = response[y2 * w + x2];
        }
    std::vector<uint8_t> zc(h * w, 0);
    for (int p = 0; p < h * w; ++p)
        if (nb1[p] * nb2[p] < 0.0) zc[p] = 1;

    std::string pre = argv[2];
    write_f64(pre + ".response.f64", response, h, w);
    write_f64(pre + ".d.f64", d, h, w);
    write_f64(pre + ".e.f64", e, h, w);
    write_f64(pre + ".zc.f64", std::vector<double>(zc.begin(), zc.end()), h, w);
    std::printf("done %s: %dx%d sigma=%.2f sigma_w=%.2f zc_px=%zu\n",
                argv[1], w, h, sigma, sigma_w,
                (size_t)std::count(zc.begin(), zc.end(), 1));
    return 0;
}
