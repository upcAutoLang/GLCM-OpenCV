/*=================================================================
 * Calculate GLCM(Gray-level Co-occurrence Matrix) By OpenCV.
 *
 * Copyright (C) 2017 Chandler Geng. All rights reserved.
 *
 *     This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 *     You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc., 59
 * Temple Place, Suite 330, Boston, MA 02111-1307 USA
===================================================================
*/
#ifndef GLCM_H
#define GLCM_H

#include "cv.h"
#include "highgui.h"
#include <math.h>

using namespace cv;
using namespace std;

// 灰度等级
// Gray Level (Choose in 4/8/16)
enum GrayLevel
{
    GRAY_4,
    GRAY_8,
    GRAY_16
};

// 灰度统计方向
// Gray Value Statistical Direction
// (Choose in 0°, 45°, 90°, 135°)
enum GrayDirection
{
    DIR_0,
    DIR_45,
    DIR_90,
    DIR_135
};

// 彩色图中的指定通道
// Point out R, G, B Channel of a Image
enum RGBChannel
{
    CHANNEL_R,
    CHANNEL_G,
    CHANNEL_B
};

// 纹理特征值结构体
// struct including Texture Eigenvalues
struct TextureEValues
{
    // 能量
    float energy;
    // 对比度
    float contrast;
    // 相关度
    float homogenity;
    // 熵
    float entropy;
};

class GLCM
{
public:
    // 从彩色通道中提取一个通道
    // Extract a channel from RGB Image
    void getOneChannel(Mat src, Mat& dstChannel, RGBChannel channel = CHANNEL_R);

    // 将灰度图中的所有像素值量级化，可以被量化为4/8/16个等级
    // Magnitude all pixels of Gray Image, and Magnitude Level can be chosen in 4/8/16;
    void GrayMagnitude(Mat src, Mat& dst, GrayLevel level = GRAY_8);

    // 计算一个矩阵窗口中，按照某个方向统计的灰度共生矩阵
    // Calculate the GLCM of one Mat Window according to one Statistical Direction.
    void CalcuOneGLCM(Mat src, Mat &dst, int src_i, int src_j, int size, GrayLevel level = GRAY_8, GrayDirection direct = DIR_0);

    // 矩阵的归一化，将矩阵所有元素与矩阵中所有元素之和作除运算，得到概率矩阵
    //   Normalize the Martix, make all pixels of Mat divided by the sum of all pixels of Mat, then get Probability Matrix.
    void NormalizeMat(Mat src, Mat& dst);

    // 计算单个窗口矩阵的图像纹理特征值，包括能量、对比度、相关度、熵
    // Calculate Texture Eigenvalues of One Window Mat, which is including Energy, Contrast, Homogenity, Entropy.
    void CalcuOneTextureEValue(Mat src, TextureEValues& EValue, bool ToCheckMat = false);

    // 计算全图的图像纹理特征值，包括能量、对比度、相关度、熵
    // Calculate Texture Eigenvalues of One Window Mat, which is including Energy, Contrast, Homogenity, Entropy.
    void CalcuTextureEValue(Mat src, TextureEValues& EValue,
                            int size = 5, GrayLevel level = GRAY_8);

    // 计算整幅图像的纹理特征
    void CalcuTextureImages(Mat src, Mat& imgEnergy, Mat& imgContrast, Mat& imgHomogenity, Mat& imgEntropy,
                            int size = 5, GrayLevel level = GRAY_8, bool ToAdjustImg = false);
};

#endif // GLCM_H
