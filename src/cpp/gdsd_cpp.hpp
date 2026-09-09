// =====================================================================
// gdsd_cpp.hpp — GDSD native C++ core (bit-exact same algorithm as gdsd.py)
// =====================================================================
#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>

namespace gdsd {

struct EdgeResult {
    std::vector<uint8_t> edge;      // h*w, 0/1
    std::vector<double> gradient_magnitude; // h*w
    std::vector<double> response;   // h*w
    std::vector<double> d;          // linear coefficient along x (gradient), h*w
    std::vector<double> e;          // linear coefficient along y (gradient), h*w
    double Tp = 0.0;
    double Tlow = 0.0;
};

// Matches gdsd.py: pinv of the design matrix (2r+1)^2 x 6
inline std::vector<std::vector<double>> design_pinv(int radius, double step) {
    int win = 2 * radius + 1;
    int N = win * win;
    // build A (N x 6)
    std::vector<std::vector<double>> A(N, std::vector<double>(6));
    int idx = 0;
    for (int r = -radius; r <= radius; ++r) {
        for (int c = -radius; c <= radius; ++c) {
            double x = c * step, y = r * step;
            A[idx][0] = x * x; A[idx][1] = y * y; A[idx][2] = x * y;
            A[idx][3] = x;     A[idx][4] = y;     A[idx][5] = 1.0;
            ++idx;
        }
    }
    // pinv = (A^T A)^{-1} A^T  (A full column rank 6)
    // compute AtA (6x6) and At (6xN)
    std::vector<std::vector<double>> AtA(6, std::vector<double>(6, 0.0));
    std::vector<std::vector<double>> At(6, std::vector<double>(N, 0.0));
    for (int i = 0; i < 6; ++i)
        for (int n = 0; n < N; ++n)
            At[i][n] = A[n][i];
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            for (int n = 0; n < N; ++n)
                AtA[i][j] += At[i][n] * A[n][j];
    // invert AtA via Gauss-Jordan
    std::vector<std::vector<double>> inv(6, std::vector<double>(6, 0.0));
    for (int i = 0; i < 6; ++i) inv[i][i] = 1.0;
    for (int col = 0; col < 6; ++col) {
        int piv = col;
        for (int r = col + 1; r < 6; ++r)
            if (std::fabs(AtA[r][col]) > std::fabs(AtA[piv][col])) piv = r;
        std::swap(AtA[col], AtA[piv]);
        std::swap(inv[col], inv[piv]);
        double d = AtA[col][col];
        for (int j = 0; j < 6; ++j) { AtA[col][j] /= d; inv[col][j] /= d; }
        for (int r = 0; r < 6; ++r) {
            if (r == col) continue;
            double f = AtA[r][col];
            for (int j = 0; j < 6; ++j) { AtA[r][j] -= f * AtA[col][j]; inv[r][j] -= f * inv[col][j]; }
        }
    }
    // pinv = inv * At  (6 x N)
    std::vector<std::vector<double>> pinv(6, std::vector<double>(N, 0.0));
    for (int i = 0; i < 6; ++i)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < 6; ++k)
                pinv[i][n] += inv[i][k] * At[k][n];
    return pinv;
}

// Matches gdsd.py to_gray_float: BGR -> gray float [0,1] (divide by 255 if max>1)
// input: uint8 HxWx3 BGR or HxW gray
inline void to_gray_float(const std::vector<uint8_t>& img, int h, int w, int channels,
                          std::vector<double>& gray) {
    gray.resize(h * w);
    if (channels == 3) {
        for (int i = 0; i < h * w; ++i)
            gray[i] = (0.114 * img[3*i] + 0.587 * img[3*i+1] + 0.299 * img[3*i+2]) / 255.0;
    } else {
        for (int i = 0; i < h * w; ++i) gray[i] = img[i] / 255.0;
    }
}

// Gaussian blur (kernel size derived from sigma via cv::Size(0,0), like
// Python cv2.GaussianBlur (0,0)) — uses the real cv::GaussianBlur (OpenCV)
// so results match Python exactly; a fixed 5x5 kernel saturates at eff. sigma ~1.38
inline void gaussian_blur(const std::vector<double>& src, int h, int w,
                          double sigma, std::vector<double>& dst) {
    dst.resize(h * w);
    // src is float [0,1] -> CV_64F Mat
    cv::Mat src_mat(h, w, CV_64F, (void*)src.data());
    cv::Mat blur_mat;
    cv::GaussianBlur(src_mat, blur_mat, cv::Size(0, 0), sigma, sigma,
                     cv::BORDER_REFLECT);
    std::memcpy(dst.data(), blur_mat.data, h * w * sizeof(double));
}

// fit_quadratic_maps + detect_edges as a whole (matches gdsd.py detect_edges)
// takes gray float [0,1] of size h x w
inline EdgeResult detect_edges(const std::vector<double>& gray, int h, int w,
                               double percentile, double sigma,
                               double gradient_threshold, int radius, double step) {
    // blur
    std::vector<double> blurred;
    gaussian_blur(gray, h, w, sigma, blurred);

    int win = 2 * radius + 1;
    // design_pinv depends only on (radius, step) — compute once, cache across images
    static const auto pinv = design_pinv(radius, step);  // 6 x N

    // padded (BORDER_REFLECT like np.pad mode="edge")
    std::vector<double> pad((h + 2*radius) * (w + 2*radius));
    auto pat = [&](int y, int x) -> double {
        int yy = std::min(std::max(y, 0), h-1);
        int xx = std::min(std::max(x, 0), w-1);
        return blurred[yy * w + xx];
    };
    for (int y = -radius; y < h + radius; ++y)
        for (int x = -radius; x < w + radius; ++x)
            pad[(y+radius) * (w + 2*radius) + (x+radius)] = pat(y, x);

    // coefficient maps a,b,c,d,e,f
    std::vector<double> a(h*w), b(h*w), c(h*w), d(h*w), e(h*w), f(h*w);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double patch[25];
            int idx = 0;
            for (int dy = 0; dy < win; ++dy)
                for (int dx = 0; dx < win; ++dx)
                    patch[idx++] = pad[(y+dy) * (w + 2*radius) + (x+dx)];
            double coef[6] = {0,0,0,0,0,0};
            for (int i = 0; i < 6; ++i) {
                double acc = 0.0;
                for (int n = 0; n < win*win; ++n) acc += pinv[i][n] * patch[n];
                coef[i] = acc;
            }
            int p = y * w + x;
            a[p] = coef[0]; b[p] = coef[1]; c[p] = coef[2];
            d[p] = coef[3]; e[p] = coef[4]; f[p] = coef[5];
        }
    }

    EdgeResult res;
    res.gradient_magnitude.resize(h*w);
    res.response.resize(h*w);
    res.d.resize(h*w);
    res.e.resize(h*w);
    res.edge.resize(h*w, 0);
    std::vector<uint8_t> valid(h*w, 0);
    std::vector<double> resp_valid;
    resp_valid.reserve(h*w);

    for (int p = 0; p < h*w; ++p) {
        double g = std::hypot(d[p], e[p]);
        res.gradient_magnitude[p] = g;
        res.d[p] = d[p];
        res.e[p] = e[p];
        double R = 2.0*a[p]*d[p]*d[p] + 2.0*c[p]*d[p]*e[p] + 2.0*b[p]*e[p]*e[p];
        res.response[p] = R;
        if (g > gradient_threshold) { valid[p] = 1; resp_valid.push_back(R); }
    }

    // percentile — O(n) nth_element (matches np.percentile: linear interpolation
    // of sorted values; nth_element finds the 2 positions instead of sorting all)
    auto percentile_val = [](std::vector<double>& v, double pct) {
        if (v.empty()) return 0.0;
        size_t n = v.size();
        if (n == 1) return v[0];
        double rank = (pct / 100.0) * (n - 1);
        size_t lo = (size_t)std::floor(rank);
        size_t hi = (size_t)std::ceil(rank);
        if (lo == hi) {
            std::nth_element(v.begin(), v.begin() + lo, v.end());
            return v[lo];
        }
        std::nth_element(v.begin(), v.begin() + lo, v.end());
        double vlo = v[lo];
        std::nth_element(v.begin(), v.begin() + hi, v.end());
        double vhi = v[hi];
        double frac = rank - lo;
        return vlo * (1.0 - frac) + vhi * frac;
    };
    double tp = percentile_val(resp_valid, percentile);
    double tlow = percentile_val(resp_valid, 100.0 - percentile);
    res.Tp = tp; res.Tlow = tlow;

    // neighbors along the quantized gradient direction + zero crossing + percentile contrast
    std::vector<double> nb1(h*w, 0.0), nb2(h*w, 0.0);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int p = y * w + x;
            if (!valid[p]) continue;
            double dx_ = d[p], dy_ = e[p];
            double mag = std::hypot(dx_, dy_);
            if (mag <= 1e-12) continue;
            // np.round = banker's rounding (half-to-even) -> std::nearbyint
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
    }
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < h*w; ++p) {
        if (!valid[p]) continue;
        bool pcontrast = ((nb1[p] > tp) && (nb2[p] < tlow)) || ((nb2[p] > tp) && (nb1[p] < tlow));
        bool zc = nb1[p] * nb2[p] < 0.0;
        if (pcontrast && zc) res.edge[p] = 1;
    }

    return res;
}

} // namespace gdsd
