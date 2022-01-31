#ifndef __LPQ_H__
#define __LPQ_H__

#include <opencv2/core.hpp>

namespace lpq {

typedef enum {
    FREQ_ESTIM_UNIFORM_WINDOW,
    FREQ_ESTIM_GAUSSIAN_WINDOW,
    FREQ_ESTIM_GAUSSIAN_DERIVATIVE,
} estimation_t;

typedef enum  { 
    CONVOLUTION_FULL,
    CONVOLUTION_SAME,
    CONVOLUTION_VALID,
} convolution_t;

class Ctx {
public:
    const double rho = 0.9;
    const double PI = 3.1415926535897932385;
    const convolution_t convmode = CONVOLUTION_VALID;
    cv::Mat w0;
    cv::Mat w1;
    cv::Mat w2;
    cv::Mat Vt;

    void setup(int winSize = 3, estimation_t freqEstim = FREQ_ESTIM_UNIFORM_WINDOW);

private:
    void CalcW(int winSize, estimation_t freqEstim);
    void CalcW_UniformWindow(const cv::Mat &x, double STFTalpha);
    void CalcW_GuassianWindow(const cv::Mat &x, double STFTalpha, double sigmaS);
    void CalcW_GuassianDerivative(const cv::Mat &x, const cv::Mat &u, double sigmaA);
    void CalcV(int winSize);
};

/* 输入矩阵元素类型需为 CV_32FC1，其他类型需要修改初始化函数 */
cv::Mat lpq(const cv::Mat &face, const Ctx &ctx, bool decorr = true);

};

#endif
