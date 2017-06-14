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

#include "glcm.h"

/*===================================================================
 * 函数名：getOneChannel
 * 说明：从彩色通道中提取一个通道；
 * 参数：
 *   Mat src:  源图像
 *   Mat& dstChannel:  源图像中的单一通道，并输出为灰度图
 *   RGBChannel channel:  RGB通道中的指定通道
 * 返回值：void
 *------------------------------------------------------------------
 * Function: getOneChannel
 *
 * Summary:
 *   Extract a channel from RGB Image;
 *
 * Arguments:
 *   Mat src - source image
 *   Mat& dstChannel - a channel from RGB source image
 *   RGBChannel channel - Point out which channel will be extracted
 *
 * Returns:
 *   void
=====================================================================
*/
void GLCM::getOneChannel(Mat src, Mat& dstChannel, RGBChannel channel)
{
    // 若输入图像已经是灰度图，则直接输出
    if(src.channels() == 1)
        dstChannel = src;

    vector<Mat> bgr;
    // 分离图像
    split(src, bgr);

    switch(channel)
    {
    case CHANNEL_B: dstChannel = bgr[0]; break;
    case CHANNEL_G: dstChannel = bgr[1]; break;
    case CHANNEL_R: dstChannel = bgr[2]; break;
    default:
        cout<<"ERROR in getOneChannel(): No Such Channel."<<endl;
        return ;
    }
}

/*===================================================================
 * 函数名：GrayMagnitude
 * 说明：将灰度图中的所有像素值量级化，可以被量化为4/8/16个等级
 * 参数：
 *   Mat src:  源图像
 *   Mat& dst:  目标图像
 *   GrayLevel level:  灰度等级
 * 返回值：void
 *------------------------------------------------------------------
 * Function: GrayMagnitude
 *
 * Summary:
 *   Magnitude all pixels of Gray Image, and Magnitude Level can be
 * chosen in 4/8/16;
 *
 * Arguments:
 *   Mat src - source image
 *   Mat& dst - destination image
 *   GrayLevel level - Destination image's Gray Level (choose in 4/8/16)
 *
 * Returns:
 *   void
=====================================================================
*/
void GLCM::GrayMagnitude(Mat src, Mat& dst, GrayLevel level)
{
    Mat tmp;
    src.copyTo(tmp);
    if(tmp.channels() == 3)
        cvtColor(tmp, tmp, CV_BGR2GRAY);

    // 直方图均衡化
    // Equalize Histogram
    equalizeHist(tmp, tmp);

    for(int j = 0; j < tmp.rows; j++)
    {
        const uchar* current = tmp.ptr<uchar>(j);
        uchar* output = dst.ptr<uchar>(j);

        for(int i = 0; i < tmp.cols; i++)
        {
            switch(level)
            {
            case GRAY_4:
                output[i] = cv::saturate_cast<uchar>(current[i] / 64);
                break;
            case GRAY_8:
                output[i] = cv::saturate_cast<uchar>(current[i] / 32);
                break;
            case GRAY_16:
                output[i] = cv::saturate_cast<uchar>(current[i] / 16);
                break;
            default:
                cout<<"ERROR in GrayMagnitude(): No Such GrayLevel."<<endl;
                return ;
            }
        }
    }
}

/*===================================================================
 * 函数名：CalcuOneGLCM
 * 说明：计算一个矩阵窗口中，按照某个方向统计的灰度共生矩阵
 * 参数：
 *   Mat src:  源图像
 *   Mat& dst:  目标灰度共生矩阵，根据选择的灰度等级，尺寸为4*4, 8*8, 16*16
 *   int src_i:  矩阵窗口中心点的行值
 *   int src_j:  矩阵窗口中心点的列值
 *   int size:  窗口尺寸（仅支持5*5, 7*7）
 *   GrayLevel level:  灰度等级
 *   GrayDirection direct:  统计方向
 * 返回值：void
 *------------------------------------------------------------------
 * Function: CalcuOneGLCM
 *
 * Summary:
 *   Calculate the GLCM of one Mat Window according to one Statistical
 * Direction.
 *
 * Arguments:
 *   Mat src - source image
 *   Mat& dst - destination GLCM, whose size is 4*4, 8*8, 16*16 by chosen
 * Gray Level
 *   int src_i - row number of Mat Window's Center Point
 *   int src_j - col number of Mat Window's Center Point
 *   int size - size of Mat Window (only support 5*5, 7*7)
 *   GrayLevel level - Destination image's Gray Level (choose in 4/8/16)
 *   GrayDirection direct - Statistical Direction (Choose in 0, 45, 90, 135)
 *
 * Returns:
 *   void
=====================================================================
*/
void GLCM::CalcuOneGLCM(Mat src, Mat& dst, int src_i, int src_j, int size, GrayLevel level, GrayDirection direct)
{
    // 灰度共生矩阵
    // GLCM
    Mat glcm;

    // 窗口矩阵
    // Window Matrix
    Mat srcCut;

    // 原图像尺寸判断
    // Judge the Size of Source Image
    if(src.cols <= 0 || src.rows <= 0)
    {
        cout<<"ERROR in CalcuOneGLCM(): source Mat's size is smaller than 0."<<endl;
        return ;
    }

    // 强制将尺寸转为奇数
    // Force Changing Window Size into odd number
    size = size / 2 * 2 + 1;

    // 边缘部分创建窗口矩阵
    // Create Mat Window for the Edges of source image
    if(src_i + (size/2) + 1 > src.rows
            || src_j + (size/2) + 1 > src.cols
            || src_i < (size/2)
            || src_j < (size/2))
    {
        size = 3;
        if(src_i <= size/2)
        {
            if(src_j <= size/2)
                srcCut = Mat(src, Range(0, 3), Range(0, 3));
            else if(src_j + (size/2) + 1 > src.cols)
                srcCut = Mat(src, Range(0, 3), Range(src.cols - 3, src.cols));
            else
                srcCut = Mat(src, Range(0, 3), Range(src_j - size/2, src_j + size/2 + 1));
        }
        else if(src_i >= src.rows - size/2)
        {
            if(src_j <= size/2)
                srcCut = Mat(src, Range(src.rows - 3, src.rows), Range(0, 3));
            else if(src_j + (size/2) + 1 > src.cols)
                srcCut = Mat(src, Range(src.rows - 3, src.rows), Range(src.cols - 3, src.cols));
            else
                srcCut = Mat(src, Range(src.rows - 3, src.rows), Range(src_j - size/2, src_j + size/2 + 1));
        }
        else if(src_j <= size/2)
        {
            if(src_i <= size/2)
                srcCut = Mat(src, Range(0, 3), Range(0, 3));
            else if(src_i + (size/2) + 1 > src.rows)
                srcCut = Mat(src, Range(src.rows - 3, src.rows), Range(0, 3));
            else
                srcCut = Mat(src, Range(src_i - size/2, src_i + size/2 + 1), Range(0, 3));
        }
        else if(src_j >= src.cols - size/2)
        {
            if(src_i <= size/2)
                srcCut = Mat(src, Range(0, 3), Range(src.cols - 3, src.cols));
            else if(src_i + (size/2) + 1 > src.rows)
                srcCut = Mat(src, Range(src.rows - 3, src.rows), Range(src.cols - 3, src.cols));
            else
                srcCut = Mat(src, Range(src_i - size/2, src_i + size/2 + 1), Range(src.cols - 3, src.cols));
        }
        else
            srcCut = Mat(src, Range(src_i - size/2, src_i + size/2 + 1), Range(src_j - size/2, src_j + size/2 + 1));
    }
    else
        srcCut = Mat(src, Range(src_i - size/2, src_i + size/2 + 1), Range(src_j - size/2, src_j + size/2 + 1));

    // 根据灰度等级初始化灰度共生矩阵
    // Initialize GLCM according Gray Level
    switch(level)
    {
    case GRAY_4:
    {
        glcm = Mat_<uchar>(4, 4);
        for(int i = 0; i < 4; i++)
            for(int j = 0; j < 4; j++)
                glcm.at<uchar>(j, i) = 0;
        break;
    }
    case GRAY_8:
    {
        glcm = Mat_<uchar>(8, 8);
        for(int i = 0; i < 8; i++)
            for(int j = 0; j < 8; j++)
                glcm.at<uchar>(j, i) = 0;
        break;
    }
    case GRAY_16:
    {
        glcm = Mat_<uchar>(16, 16);
        for(int i = 0; i < 16; i++)
            for(int j = 0; j < 16; j++)
                glcm.at<uchar>(j, i) = 0;
        break;
    }
    default:
        cout<<"ERROR in CalcuOneGLCM(): No Such Gray Level."<<endl;
        break;
    }

    // 根据统计方向填充灰度共生矩阵
    // Fill GLCM according Statistical Direction
    switch(direct)
    {
    case DIR_0:
        for(int i = 0; i < srcCut.rows; i++)
            for(int j = 0; j < srcCut.cols - 1; j++)
                glcm.at<uchar>(srcCut.at<uchar>(j, i), srcCut.at<uchar>(j+1, i))++;
        break;
    case DIR_45:
        for(int i = 0; i < srcCut.rows - 1; i++)
            for(int j = 0; j < srcCut.cols - 1; j++)
                glcm.at<uchar>(srcCut.at<uchar>(j, i), srcCut.at<uchar>(j+1, i+1))++;
        break;
    case DIR_90:
        for(int i = 0; i < srcCut.rows - 1; i++)
            for(int j = 0; j < srcCut.cols; j++)
                glcm.at<uchar>(srcCut.at<uchar>(j, i), srcCut.at<uchar>(j, i+1))++;
        break;
    case DIR_135:
        for(int i = 1; i < srcCut.rows; i++)
            for(int j = 0; j < srcCut.cols - 1; j++)
                glcm.at<uchar>(srcCut.at<uchar>(j, i), srcCut.at<uchar>(j+1, i-1))++;
        break;
    default:
        cout<<"ERROR in CalcuOneGLCM(): No such Direct."<<endl;
        break;
    }

    Mat glcm_dst;
    // 灰度共生矩阵归一化
    // Normalize GLCM
    NormalizeMat(glcm, glcm_dst);
    glcm_dst.copyTo(dst);
}

/*===================================================================
 * 函数名：NormalizeMat
 * 说明：矩阵的归一化，将矩阵所有元素与矩阵中所有元素之和作除运算，得到概率矩阵
 * 参数：
 *   Mat src:  源图像
 *   Mat& dst:  目标概率矩阵
 * 返回值：void
 *------------------------------------------------------------------
 * Function: NormalizeMat
 *
 * Summary:
 *   Normalize the Martix, make all pixels of Mat divided by the sum of
 * all pixels of Mat, then get Probability Matrix.
 *
 * Arguments:
 *   Mat src - source image
 *   Mat& dst - destination Probability Matrix
 *
 * Returns:
 *   void
=====================================================================
*/
void GLCM::NormalizeMat(Mat src, Mat& dst)
{
    Mat tmp;
    src.convertTo(tmp, CV_32F);

    float sum = 0;
    for(int i = 0; i < tmp.rows; i++)
        for(int j = 0; j < tmp.cols; j++)
            sum += tmp.at<float>(j, i);
    if(sum == 0)    sum = 1;

    for(int i = 0; i < tmp.rows; i++)
        for(int j = 0; j < tmp.cols; j++)
            tmp.at<float>(j, i) /= sum;

    tmp.copyTo(dst);
}

/*===================================================================
 * 函数名：CalcuOneTextureEValue
 * 说明：计算单个窗口矩阵的图像纹理特征值，包括能量、对比度、相关度、熵
 * 参数：
 *   Mat src:  源矩阵，窗口矩阵
 *   TextureEValues& EValue:  纹理特征值变量
 *   bool ToCheckMat:  检查输入矩阵是否为概率矩阵
 * 返回值：void
 *------------------------------------------------------------------
 * Function: CalcuOneTextureEValue
 *
 * Summary:
 *   Calculate Texture Eigenvalues of the Window Mat, which is including
 * Energy, Contrast, Homogenity, Entropy.
 *
 * Arguments:
 *   Mat src - source Matrix (Window Mat)
 *   TextureEValues& EValue - Texture Eigenvalues
 *   bool ToCheckMat - to check input Mat is Probability Mat or not
 *
 * Returns:
 *   void
=====================================================================
*/
void GLCM::CalcuOneTextureEValue(Mat src, TextureEValues& EValue, bool ToCheckMat)
{
    if(ToCheckMat)
    {
        float sum = 0;
        for(int i = 0; i < src.rows; i++)
            for(int j = 0; j < src.cols; j++)
                sum += src.at<float>(j, i);
        if(sum < 0.99 || sum > 1.01)
        {
            cout<<"ERROR in CalcuOneTextureEValue(): Sum of the Mat is not equal to 1.00."<<endl;
            return ;
        }
    }

    EValue.contrast = 0;
    EValue.energy = 0;
    EValue.entropy = 0;
    EValue.homogenity = 0;

    for(int i = 0; i < src.rows; i++)
        for(int j = 0; j < src.cols; j++)
        {
            EValue.energy += powf(src.at<float>(j, i), 2);
            EValue.contrast += (powf((i - j), 2) * src.at<float>(j, i) );
            EValue.homogenity += (src.at<float>(j, i) / (1 + fabs((float)(i - j))) );
            if(src.at<float>(j, i) != 0)
                EValue.entropy -= (src.at<float>(j, i) * log10(src.at<float>(j, i)) );
        }
}

/*===================================================================
 * 函数名：CalcuTextureEValue
 * 说明：计算全图的图像纹理特征值，包括能量、对比度、相关度、熵
 * 参数：
 *   Mat src:  源矩阵，窗口矩阵
 *   TextureEValues& EValue:  输出目标，全图的纹理特征值变量
 *   int size:  窗口尺寸（仅支持5*5, 7*7）
 *   GrayLevel level:  灰度等级
 * 返回值：void
 *------------------------------------------------------------------
 * Function: CalcuOneTextureEValue
 *
 * Summary:
 *   Calculate Texture Eigenvalues of One Window Mat, which is including
 * Energy, Contrast, Homogenity, Entropy.
 *
 * Arguments:
 *   Mat src - source Matrix (Window Mat)
 *   TextureEValues& EValue - Output Dst: Texture Eigenvalues of the Whole Image
 *   int size - size of Mat Window (only support 5*5, 7*7)
 *   GrayLevel level - Destination image's Gray Level (choose in 4/8/16)
 *
 * Returns:
 *   void
=====================================================================
*/
void GLCM::CalcuTextureEValue(Mat src, TextureEValues& EValue, int size, GrayLevel level)
{
    // 原图像的灰度图
    // Gray Image of the Source Image
    Mat imgGray;

    // 窗口矩阵
    // Window Matrix
    Mat glcm_win;

    // 归一化后的概率矩阵
    // Probability Matrix after Normalizing
    Mat glcm_norm;

    // 纹理特征值缓存变量
    // Texture Eigenvalues temp variable
    TextureEValues EValue_temp;

    // 初始化目标纹理特征值
    // Init Dst Texture Eigenvalues
    EValue.contrast = 0; EValue.energy = 0; EValue.entropy = 0; EValue.homogenity = 0;

    // 检查输入图像是否为单通道图像，如果不是，则转换其格式
    // Check if Input Image is Single Channel Image or not, IF it's Single Channel Image, then Convert its Format to Gray Image.
    if(src.channels() != 1)
        cvtColor(src, imgGray, CV_BGR2GRAY);
    else
        src.copyTo(imgGray);

    for(int i = 0; i < imgGray.rows; i++)
    {
        for(int j = 0; j < imgGray.cols; j++)
        {
            // 计算所有统计方向的灰度共生矩阵与对应的特征值，并累加至缓存变量中
            // Calculate All Statistical Direction's GLCM and Eigenvalues, then accumulate into temp variables
            float energy, contrast, homogenity, entropy;
            energy = contrast = homogenity = entropy = 0;

            CalcuOneGLCM(imgGray, glcm_win, i, j, size, level, DIR_0);
            NormalizeMat(glcm_win, glcm_norm);
            CalcuOneTextureEValue(glcm_norm, EValue_temp, false);
            energy += EValue_temp.energy; contrast += EValue_temp.contrast;
            homogenity += EValue_temp.homogenity; entropy += EValue_temp.entropy;

            CalcuOneGLCM(imgGray, glcm_win, i, j, size, level, DIR_45);
            NormalizeMat(glcm_win, glcm_norm);
            CalcuOneTextureEValue(glcm_norm, EValue_temp, false);
            energy += EValue_temp.energy; contrast += EValue_temp.contrast;
            homogenity += EValue_temp.homogenity; entropy += EValue_temp.entropy;

            CalcuOneGLCM(imgGray, glcm_win, i, j, size, level, DIR_90);
            NormalizeMat(glcm_win, glcm_norm);
            CalcuOneTextureEValue(glcm_norm, EValue_temp, false);
            energy += EValue_temp.energy; contrast += EValue_temp.contrast;
            homogenity += EValue_temp.homogenity; entropy += EValue_temp.entropy;

            CalcuOneGLCM(imgGray, glcm_win, i, j, size, level, DIR_135);
            NormalizeMat(glcm_win, glcm_norm);
            CalcuOneTextureEValue(glcm_norm, EValue_temp, false);
            energy += EValue_temp.energy; contrast += EValue_temp.contrast;
            homogenity += EValue_temp.homogenity; entropy += EValue_temp.entropy;

            // 将所有方向计算得到的特征值平均化，得到的值即可消除统计方向影响
            // average Eigenvalues of all Statistical Directions, then the average value has eliminated the effect of Statistical Directions
            energy /= 4; contrast /= 4;
            homogenity /= 4; entropy /= 4;

            // 累加当前单个窗口的纹理特征值，作为整个图像的纹理特征值
            // Accumulate Texture Eigenvalues of Current Window, then make the Sum as Texture Eigenvalues of the Whole Image
            EValue.contrast += contrast;
            EValue.energy += energy;
            EValue.entropy += entropy;
            EValue.homogenity += homogenity;
        }
    }
}

/*===================================================================
 * 函数名：CalcuTextureImages
 * 说明：计算整幅图像的纹理特征，并将结果输出到相应矩阵中
 * 参数：
 *   Mat src:  原图像
 *   Mat& imgEnergy:  目标能量矩阵
 *   Mat& imgContrast:  目标对比度矩阵
 *   Mat& imgHomogenity:  目标相关度矩阵
 *   Mat& imgEntropy:  目标熵矩阵
 *   int size:  窗口尺寸（仅支持5*5, 7*7）
 *   GrayLevel level:  灰度等级
 *   bool ToAdjustImg:  是否调整输出的纹理特征图像
 * 返回值：void
 *------------------------------------------------------------------
 * Function: CalcuTextureImages
 *
 * Summary:
 *   Calculate Texture Features of the whole Image, and output the result
 * into Martixs.
 *
 * Arguments:
 *   Mat src - source Image
 *   Mat& imgEnergy - Destination Mat, Energy Matrix
 *   Mat& imgContrast - Destination Mat, Contrast Matrix
 *   Mat& imgHomogenity - Destination Mat, Homogenity Matrix
 *   Mat& imgEntropy - Destination Mat, Entropy Matrix
 *   int size - size of Mat Window (only support 5*5, 7*7)
 *   GrayLevel level - Destination image's Gray Level (choose in 4/8/16)
 *   bool ToAdjustImg:  to Adjust output Texture Feature Images or not
 *
 * Returns:
 *   void
=====================================================================
*/
void GLCM::CalcuTextureImages(Mat src, Mat& imgEnergy, Mat& imgContrast, Mat& imgHomogenity, Mat& imgEntropy,
                        int size, GrayLevel level, bool ToAdjustImg)
{
    // 窗口矩阵
    // Window Matrix
    Mat glcm_win;

    // 归一化后的概率矩阵
    // Probability Matrix after Normalizing
    Mat glcm_norm;

    // 纹理特征值缓存变量
    // Texture Eigenvalues temp varialbe
    TextureEValues EValue;

    imgEnergy.create(src.size(), CV_32FC1);
    imgContrast.create(src.size(), CV_32FC1);
    imgHomogenity.create(src.size(), CV_32FC1);
    imgEntropy.create(src.size(), CV_32FC1);

    for(int i = 0; i < src.rows; i++)
    {
        float* energyData = imgEnergy.ptr<float>(i);
        float* contrastData = imgContrast.ptr<float>(i);
        float* homogenityData = imgHomogenity.ptr<float>(i);
        float* entropyData = imgEntropy.ptr<float>(i);

        for(int j = 0; j < src.cols; j++)
        {
            // 计算所有统计方向的灰度共生矩阵与对应的特征值，并累加至缓存变量中
            // Calculate All Statistical Direction's GLCM and Eigenvalues, then accumulate into temp variables
            float energy, contrast, homogenity, entropy;
            energy = contrast = homogenity = entropy = 0;

            CalcuOneGLCM(src, glcm_win, i, j, size, level, DIR_0);
            NormalizeMat(glcm_win, glcm_norm);
            CalcuOneTextureEValue(glcm_norm, EValue, false);
            energy += EValue.energy; contrast += EValue.contrast;
            homogenity += EValue.homogenity; entropy += EValue.entropy;

            CalcuOneGLCM(src, glcm_win, i, j, size, level, DIR_45);
            NormalizeMat(glcm_win, glcm_norm);
            CalcuOneTextureEValue(glcm_norm, EValue, false);
            energy += EValue.energy; contrast += EValue.contrast;
            homogenity += EValue.homogenity; entropy += EValue.entropy;

            CalcuOneGLCM(src, glcm_win, i, j, size, level, DIR_90);
            NormalizeMat(glcm_win, glcm_norm);
            CalcuOneTextureEValue(glcm_norm, EValue, false);
            energy += EValue.energy; contrast += EValue.contrast;
            homogenity += EValue.homogenity; entropy += EValue.entropy;

            CalcuOneGLCM(src, glcm_win, i, j, size, level, DIR_135);
            NormalizeMat(glcm_win, glcm_norm);
            CalcuOneTextureEValue(glcm_norm, EValue, false);
            energy += EValue.energy; contrast += EValue.contrast;
            homogenity += EValue.homogenity; entropy += EValue.entropy;

            // 将所有方向计算得到的特征值平均化，得到的值即可消除统计方向影响
            // average Eigenvalues of all Statistical Directions, then the average value has eliminated the effect of Statistical Directions
            energy /= 4; contrast /= 4;
            homogenity /= 4; entropy /= 4;

            energyData[j] = energy;
            contrastData[j] = contrast;
            homogenityData[j] = homogenity;
            entropyData[j] = entropy;
        }
    }

    // 调整输出特征图像，类型由CV_32FC1改为CV_8UC1，取值范围0--255
    // Adjust output Texture Feature Images, Change its type from CV_32FC1 to CV_8UC1, Change its value range as 0--255
    if(ToAdjustImg)
    {
        cv::normalize(imgEnergy, imgEnergy, 0, 255, NORM_MINMAX);
        cv::normalize(imgContrast, imgContrast, 0, 255, NORM_MINMAX);
        cv::normalize(imgEntropy, imgEntropy, 0, 255, NORM_MINMAX);
        cv::normalize(imgHomogenity, imgHomogenity, 0, 255, NORM_MINMAX);
        imgEnergy.convertTo(imgEnergy, CV_8UC1);
        imgContrast.convertTo(imgContrast, CV_8UC1);
        imgEntropy.convertTo(imgEntropy, CV_8UC1);
        imgHomogenity.convertTo(imgHomogenity, CV_8UC1);
    }
}
