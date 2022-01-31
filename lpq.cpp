#include "lpq.hpp"

#include <vector>
#include <complex>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/**
 * 本过程根据 https://github.com/jenja/Face_Recognition/blob/master/lpq.m 重构实现
 */

namespace lpq {

/* 生成元素在[a, b]区间的一维矩阵 */
static cv::Mat ComplexMatList(int a, int b)
{
    int len = b - a + 1;
    cv::Mat mat(1, len, CV_64FC2);
    cv::Mat_<cv::Complexd> helper = mat;
    for (int j = 0; j < len; j++) {
        helper(0, j) = cv::Complexd(a + j, 0);
    }
    return mat;
}

/* 将CV_32FC1矩阵转换为Complex(CV_64FC2) */
static void ComplexMatInit(cv::Mat &complex, const cv::Mat &real)
{
    complex.create(real.rows, real.cols, CV_64FC2);
    cv::Mat_<cv::Complexd> helper = complex;
    for (int i = 0; i < complex.rows; i++) {
        for (int j = 0; j < complex.cols; j++) {
            helper(i, j) = cv::Complexd(real.at<float>(i, j), 0);
        }
    }
}

/* 将Complex矩阵分拆实部和虚部 */
static void ComplexMatSplit(const cv::Mat &complex, cv::Mat &real, cv::Mat &imag)
{
    real.create(complex.rows, complex.cols, CV_64FC1);
    imag.create(complex.rows, complex.cols, CV_64FC1);

    const cv::Mat_<cv::Complexd> helper = complex;
    for (int i = 0; i < complex.rows; i++) {
        for (int j = 0; j < complex.cols; j++) {
            real.at<double>(i, j) = helper(i, j).re;
            imag.at<double>(i, j) = helper(i, j).im;
        }
    }
}

/* 将实部虚部合并Complex矩阵 */
static void ComplexMatMerge(cv::Mat &complex, const cv::Mat &real, const cv::Mat &imag)
{
    complex.create(real.rows, real.cols, CV_64FC2);
    cv::Mat_<cv::Complexd> helper = complex;
    for (int i = 0; i < complex.rows; i++) {
        for (int j = 0; j < complex.cols; j++) {
            double re = real.at<double>(i, j);
            double im = imag.at<double>(i, j);
            helper(i, j) = cv::Complexd(re, im);
        }
    }
}

/* W和V为计算LPQ特征时与原图无关的数据，可以单独提取为Ctx */
void Ctx::setup(int winSize, estimation_t freqEstim)
{
    this->CalcW(winSize, freqEstim);
    this->CalcV(winSize);
}

void Ctx::CalcW(int winSize, estimation_t freqEstim)
{
    double STFTalpha = 1.0 / winSize;
    double sigmaS = (winSize - 1) / 4.0;
    double sigmaA = 8.0 / (winSize - 1);
    int r = (winSize - 1) / 2;

    cv::Mat x = ComplexMatList(-r, r);
    cv::Mat u = ComplexMatList(1, r);

    switch (freqEstim) {
        case FREQ_ESTIM_UNIFORM_WINDOW:
            this->CalcW_UniformWindow(x, STFTalpha);
            break;
        case FREQ_ESTIM_GAUSSIAN_WINDOW:
            this->CalcW_GuassianWindow(x, STFTalpha, sigmaS);
            break;
        case FREQ_ESTIM_GAUSSIAN_DERIVATIVE:
            this->CalcW_GuassianDerivative(x, u, sigmaA);
            break;
    }
}

void Ctx::CalcW_UniformWindow(const cv::Mat &x, double STFTalpha)
{
    this->w0 = cv::Mat::ones(1, x.cols, CV_64FC2);
    this->w1.create(1, x.cols, CV_64FC2);
    this->w2.create(1, x.cols, CV_64FC2);

    cv::Mat_<cv::Complexd> hx = x;
    cv::Mat_<cv::Complexd> h1 = this->w1;
    cv::Mat_<cv::Complexd> h2 = this->w2;
    for (int j = 0; j < x.cols; j++) {
        std::complex<double> Vt(0, -2 * this->PI * hx(0, j).re * STFTalpha);
        std::complex<double> e = std::exp(Vt);
        h1(0, j) = cv::Complexd(e.real(), e.imag());
        h2(0, j) = cv::Complexd(e.real(), -e.imag());
    }
}

void Ctx::CalcW_GuassianWindow(const cv::Mat &x, double STFTalpha, double sigmaS)
{
    this->CalcW_UniformWindow(x, STFTalpha);
    // TODO
}

void Ctx::CalcW_GuassianDerivative(const cv::Mat &x, const cv::Mat &u, double sigmaA)
{
    // TODO
}

void Ctx::CalcV(int winSize)
{
    /* Matlab实现中dist在OpenCV中未找到对应的，使用一个较直接但是性能比较低的方法计算 */
    cv::Mat C(winSize * winSize, winSize * winSize, CV_64FC1);
    for (int i = 0; i < winSize * winSize; i++) {
        for (int j = 0; j < winSize * winSize; j++) {
            double dx = j / winSize - i / winSize;
            double dy = j % winSize - i % winSize;
            C.at<double>(i, j) = std::pow(this->rho, std::sqrt(dx * dx + dy * dy));
        }
    }

    cv::Mat q1 = this->w0.t() * this->w1;
    cv::Mat q2 = this->w1.t() * this->w0;
    cv::Mat q3 = this->w1.t() * this->w1;
    cv::Mat q4 = this->w1.t() * this->w2;

    cv::Mat u[8];
    ComplexMatSplit(q1, u[0], u[1]);
    ComplexMatSplit(q2, u[2], u[3]);
    ComplexMatSplit(q3, u[4], u[5]);
    ComplexMatSplit(q4, u[6], u[7]);

    cv::Mat M(8, winSize * winSize, CV_64FC1);
    for (int e = 0; e < 8; e++) {
        for (int j = 0; j < winSize; j++) {
            for (int i = 0; i < winSize; i++) {
                M.at<double>(e, i + j * winSize) = u[e].at<double>(i, j);
            }
        }
    }

    cv::Mat D = M * C * M.t();

    cv::Mat A = cv::Mat::zeros(8, 8, CV_64FC1);
    for (int i = 0; i < 8; i++) {
        A.at<double>(i, i) = 1 + 0.000001 * (8 - 1 - i);
    }

    /* Matlab的svd函数与SVD::compute函数返回值有所差异，但是只使用C++代码时并不影响LPQ的效果 */
    cv::Mat S, U;
    cv::SVD::compute(A * D * A, S, U, this->Vt);

    double maxV, minV;
    cv::Point maxL, minL;

    /* 为避免SVM模型训练时的问题，将矩阵中每列最大的值转换为正 */
    for (int i = 0; i < this->Vt.rows; i++) {
        cv::minMaxLoc(cv::abs(this->Vt.row(i)), &minV, &maxV, &minL, &maxL);
        if (this->Vt.at<double>(i, maxL.x) < 0) {
            for (int j = 0; j < this->Vt.cols; j++) {
                this->Vt.at<double>(i, j) = -this->Vt.at<double>(i, j);
            }
        }
    }
}

/* 将复数矩阵分拆为实部和虚部分开计算再合并 */
static void ComplexMatFilter2D(const cv::Mat &src, cv::Mat &dst, int depth, const cv::Mat &kernel,
    cv::Point &anchor, double delta, int borderType)
{
    cv::Mat sr, si, kr, ki;
    cv::Mat d1, d2, d3, d4;

    ComplexMatSplit(src, sr, si);
    ComplexMatSplit(kernel, kr, ki);

    cv::filter2D(sr, d1, depth, kr, anchor, delta, borderType);
    cv::filter2D(sr, d2, depth, ki, anchor, delta, borderType);
    cv::filter2D(si, d3, depth, kr, anchor, delta, borderType);
    cv::filter2D(si, d4, depth, ki, anchor, delta, borderType);

    ComplexMatMerge(dst, d1 - d4, d2 + d3);
}

/* Matlab中的conv2函数在OpenCV无对应，需要借助filter2D单独实现 */
static cv::Mat conv2(const cv::Mat &image, const cv::Mat& ikernel, convolution_t convMode)
{
    cv::Mat kernel;
    cv::flip(ikernel, kernel, -1);

    cv::Mat src;
    if (convMode == CONVOLUTION_FULL) {
        int r = kernel.rows - 1;
        int c = kernel.cols - 1;
        cv::copyMakeBorder(image, src, (r + 1) / 2, r / 2, (c + 1) / 2, c / 2, cv::BORDER_CONSTANT, cv::Scalar(0));
    } else {
        src = image;
    }

    cv::Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);

    cv::Mat dst;
    ComplexMatFilter2D(src, dst, image.depth(), kernel, anchor, 0, cv::BORDER_CONSTANT);

    if (convMode == CONVOLUTION_VALID) {
        dst = dst.colRange((kernel.cols - 1) / 2, dst.cols - kernel.cols / 2)
                 .rowRange((kernel.rows - 1) / 2, dst.rows - kernel.rows / 2);
    }

    return dst;
}

cv::Mat lpq(const cv::Mat &face, const Ctx &ctx, bool decorr)
{
    cv::Mat img;
    ComplexMatInit(img, face);

    cv::Mat freqResp[8];
    cv::Mat filterResp;

    filterResp = conv2(conv2(img, ctx.w0.t(), ctx.convmode), ctx.w1, ctx.convmode);
    ComplexMatSplit(filterResp, freqResp[0], freqResp[1]);

    filterResp = conv2(conv2(img, ctx.w1.t(), ctx.convmode), ctx.w0, ctx.convmode);
    ComplexMatSplit(filterResp, freqResp[2], freqResp[3]);

    filterResp = conv2(conv2(img, ctx.w1.t(), ctx.convmode), ctx.w1, ctx.convmode);
    ComplexMatSplit(filterResp, freqResp[4], freqResp[5]);

    filterResp = conv2(conv2(img, ctx.w1.t(), ctx.convmode), ctx.w2, ctx.convmode);
    ComplexMatSplit(filterResp, freqResp[6], freqResp[7]);

    if (decorr) {
        cv::Mat bigFreqResp(filterResp.rows * filterResp.cols, 8, CV_64FC1);
        for (int e = 0; e < 8; e++) {
            for (int j = 0; j < filterResp.cols; j++) {
                for (int i = 0; i < filterResp.rows; i++) {
                    bigFreqResp.at<double>(i + j * filterResp.rows, e) = freqResp[e].at<double>(i, j);
                }
            }
        }

        bigFreqResp = (ctx.Vt * bigFreqResp.t()).t();
        for (int e = 0; e < 8; e++) {
            for (int j = 0; j < filterResp.cols; j++) {
                for (int i = 0; i < filterResp.rows; i++) {
                    freqResp[e].at<double>(i, j) = bigFreqResp.at<double>(i + j * filterResp.rows, e);
                }
            }
        }
    }

    cv::Mat LPQdesc = cv::Mat::zeros(filterResp.rows, filterResp.cols, CV_8UC1);
    for (int e = 0; e < 8; e++) {
        unsigned char m = 1 << e;
        for (int i = 0; i < LPQdesc.rows; i++) {
            for (int j = 0; j < LPQdesc.cols; j++) {
                LPQdesc.at<unsigned char>(i, j) |= (freqResp[e].at<double>(i, j) > 0.0000000001) ? m : 0;
            }
        }
    }

    return LPQdesc;
}

};
