/*
* Copyright 2017-2020 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/**
*  This sample application illustrates encoding of frames in CUDA device buffers.
*  The application reads the image data from file and loads it to CUDA input
*  buffers obtained from the encoder using NvEncoder::GetNextInputFrame().
*  The encoder subsequently maps the CUDA buffers for encoder using NvEncodeAPI
*  and submits them to NVENC hardware for encoding as part of EncodeFrame() function.
*  The NVENC hardware output is written in system memory for this case.
*
*  This sample application also illustrates the use of video memory buffer allocated 
*  by the application to get the NVENC hardware output. This feature can be used
*  for H264 ME -only mode, H264 encode and HEVC encode. This application copies the NVENC output 
*  from video memory buffer to host memory buffer in order to dump to a file, but this
*  is not needed if application choose to use it in some other way.
*
*  Since, encoding may involve CUDA pre-processing on the input and post-processing on 
*  output, use of CUDA streams is also illustrated to pipeline the CUDA pre-processing 
*  and post-processing tasks, for output in video memory case.
*
*  CUDA streams can be used for H.264 ME-only, HEVC ME-only, H264 encode and HEVC encode.
*/

#include <fstream>
#include <iostream>
#include <memory>
#include <cuda.h>
#include "../Utils/NvCodecUtils.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "../Utils/Logger.h"
#include "../Utils/NvEncoderCLIOptions.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void ShowEncoderCapability()
{
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    std::cout << "Encoder Capability" << std::endl << std::endl;
    for (int iGpu = 0; iGpu < nGpu; iGpu++) {
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));
        NvEncoderCuda enc(cuContext, 1280, 720, NV_ENC_BUFFER_FORMAT_NV12);

        std::cout << "GPU " << iGpu << " - " << szDeviceName << std::endl << std::endl;
        std::cout << "\tH264:\t\t" << "  " <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID,
                                     NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "yes" : "no" ) << std::endl <<
            "\tH264_444:\t" << "  " <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID,
                                     NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ? "yes" : "no" ) << std::endl <<
            "\tH264_ME:\t" << "  " <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID,
                                     NV_ENC_CAPS_SUPPORT_MEONLY_MODE) ? "yes" : "no" ) << std::endl <<
            "\tH264_WxH:\t" << "  " <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID,
                                     NV_ENC_CAPS_WIDTH_MAX) ) << "*" <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_HEIGHT_MAX) ) << std::endl <<
            "\tHEVC:\t\t" << "  " <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                     NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "yes" : "no" ) << std::endl <<
            "\tHEVC_Main10:\t" << "  " <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                     NV_ENC_CAPS_SUPPORT_10BIT_ENCODE) ? "yes" : "no" ) << std::endl <<
            "\tHEVC_Lossless:\t" << "  " <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                     NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE) ? "yes" : "no" ) << std::endl <<
            "\tHEVC_SAO:\t" << "  " <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                     NV_ENC_CAPS_SUPPORT_SAO) ? "yes" : "no" ) << std::endl <<
            "\tHEVC_444:\t" << "  " <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                     NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ? "yes" : "no" ) << std::endl <<
            "\tHEVC_ME:\t" << "  " <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                     NV_ENC_CAPS_SUPPORT_MEONLY_MODE) ? "yes" : "no" ) << std::endl <<
            "\tHEVC_WxH:\t" << "  " <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                     NV_ENC_CAPS_WIDTH_MAX) ) << "*" <<
            ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_HEIGHT_MAX) ) << std::endl;

        std::cout << std::endl;

        enc.DestroyEncoder();
        ck(cuCtxDestroy(cuContext));
    }
}

void ShowHelpAndExit(const char *szBadOption = NULL)
{
    bool bThrowError = false;
    std::ostringstream oss;
    if (szBadOption) 
    {
        bThrowError = true;
        oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
    }
    oss << "Options:" << std::endl
        << "-i               Input file path" << std::endl
        << "-o               Output file path" << std::endl
        << "-s               Input resolution in this form: WxH" << std::endl
        << "-if              Input format: iyuv nv12 yuv444 p010 yuv444p16 bgra bgra10 ayuv abgr abgr10" << std::endl
        << "-gpu             Ordinal of GPU to use" << std::endl
        << "-outputInVidMem  Set this to 1 to enable output in Video Memory" << std::endl
        << "-cuStreamType    Use CU stream for pre and post processing when outputInVidMem is set to 1" << std::endl
        << "                 CRC of encoded frames will be computed and dumped to file with suffix '_crc.txt' added" << std::endl
        << "                 to file specified by -o option " << std::endl
        << "                 0 : both pre and post processing are on NULL CUDA stream" << std::endl
        << "                 1 : both pre and post processing are on SAME CUDA stream" << std::endl
        << "                 2 : both pre and post processing are on DIFFERENT CUDA stream" << std::endl
        ;
    oss << NvEncoderInitParam().GetHelpMessage() << std::endl;
    if (bThrowError)
    {
        throw std::invalid_argument(oss.str());
    }
    else
    {
        std::cout << oss.str();
        ShowEncoderCapability();
        exit(0);
    }
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, int &nWidth, int &nHeight, 
    NV_ENC_BUFFER_FORMAT &eFormat, char *szOutputFileName, NvEncoderInitParam &initParam, int &iGpu, 
    int32_t &cuStreamType)
{
    std::ostringstream oss;
    int i;
    for (i = 1; i < argc; i++)
    {
        if (!_stricmp(argv[i], "-h"))
        {
            ShowHelpAndExit();
        }
        if (!_stricmp(argv[i], "-i"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-i");
            }
            sprintf(szInputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-o"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-o");
            }
            sprintf(szOutputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-s"))
        {
            if (++i == argc || 2 != sscanf(argv[i], "%dx%d", &nWidth, &nHeight))
            {
                ShowHelpAndExit("-s");
            }
            continue;
        }
        std::vector<std::string> vszFileFormatName =
        {
            "iyuv", "nv12", "yv12", "yuv444", "p010", "yuv444p16", "bgra", "bgra10", "ayuv", "abgr", "abgr10"
        };
        NV_ENC_BUFFER_FORMAT aFormat[] = 
        {
            NV_ENC_BUFFER_FORMAT_IYUV,
            NV_ENC_BUFFER_FORMAT_NV12,
            NV_ENC_BUFFER_FORMAT_YV12,
            NV_ENC_BUFFER_FORMAT_YUV444,
            NV_ENC_BUFFER_FORMAT_YUV420_10BIT,
            NV_ENC_BUFFER_FORMAT_YUV444_10BIT,
            NV_ENC_BUFFER_FORMAT_ARGB,
            NV_ENC_BUFFER_FORMAT_ARGB10,
            NV_ENC_BUFFER_FORMAT_AYUV,
            NV_ENC_BUFFER_FORMAT_ABGR,
            NV_ENC_BUFFER_FORMAT_ABGR10,
        };
        if (!_stricmp(argv[i], "-if"))
        {
            if (++i == argc) {
                ShowHelpAndExit("-if");
            }
            auto it = std::find(vszFileFormatName.begin(), vszFileFormatName.end(), argv[i]);
            if (it == vszFileFormatName.end())
            {
                ShowHelpAndExit("-if");
            }
            eFormat = aFormat[it - vszFileFormatName.begin()];
            continue;
        }
        if (!_stricmp(argv[i], "-gpu"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-cuStreamType"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-cuStreamType");
            }
            cuStreamType = atoi(argv[i]);
            continue;
        }

        // Regard as encoder parameter
        if (argv[i][0] != '-')
        {
            ShowHelpAndExit(argv[i]);
        }
        oss << argv[i] << " ";
        while (i + 1 < argc && argv[i + 1][0] != '-')
        {
            oss << argv[++i] << " ";
        }
    }
    initParam = NvEncoderInitParam(oss.str().c_str());
}

template<class EncoderClass> 
void InitializeEncoder(EncoderClass &pEnc, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat)
{
    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };

    initializeParams.encodeConfig = &encodeConfig;
    pEnc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
    encodeCLIOptions.SetInitParams(&initializeParams, eFormat);

    pEnc->CreateEncoder(&initializeParams);
}

void EncodeCuda(int nWidth, int nHeight, NvEncoderInitParam encodeCLIOptions, CUcontext cuContext, cv::cuda::GpuMat srcIn, std::ofstream& fpOut)
{
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_ABGR;

    std::unique_ptr<NvEncoderCuda> pEnc(new NvEncoderCuda(cuContext, nWidth, nHeight, eFormat));

    InitializeEncoder(pEnc, encodeCLIOptions, eFormat);

    int nFrameSize = pEnc->GetFrameSize();
    int nFrame = 0;
    int last_frame = 15*25;
    for (int i = 0; i <= last_frame; i++)
    {

        std::streamsize nRead = nFrameSize;
        // For receiving encoded packets
        std::vector<std::vector<uint8_t>> vPacket;
        
        if (i < last_frame)
        {
            const NvEncInputFrame* encoderInputFrame = pEnc->GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(cuContext, srcIn.data, 0, (CUdeviceptr)encoderInputFrame->inputPtr,
                (int)encoderInputFrame->pitch,
                pEnc->GetEncodeWidth(),
                pEnc->GetEncodeHeight(),
                CU_MEMORYTYPE_HOST,
                encoderInputFrame->bufferFormat,
                encoderInputFrame->chromaOffsets,
                encoderInputFrame->numChromaPlanes);
            pEnc->EncodeFrame(vPacket);
        }
        else
        {
            pEnc->EndEncode(vPacket);
        }
        nFrame += (int)vPacket.size();
        for (std::vector<uint8_t>& packet : vPacket)
        {
            // For each encoded packet
            fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
        }
    }

    pEnc->DestroyEncoder();

    std::cout << "Total frames encoded: " << nFrame << std::endl;
}


int main(int argc, char **argv)
{

    char szInFilePath[256] = "",
        szOutFilePath[256] = "";
    int nWidth = 0, nHeight = 0;
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
    int iGpu = 0;
    try
    {
        NvEncoderInitParam encodeCLIOptions;
        int cuStreamType = -1;
        bool bOutputInVideoMem = false;
        ParseCommandLine(argc, argv, szInFilePath, nWidth, nHeight, eFormat, szOutFilePath, encodeCLIOptions, iGpu, 
                         cuStreamType);

        
        cv::Mat srcImgHost = cv::imread(szInFilePath);

        nWidth = srcImgHost.cols;
        nHeight = srcImgHost.rows;
        cv::cuda::GpuMat srcImgDevice;
        srcImgDevice.upload(srcImgHost);
        cv::cuda::cvtColor(srcImgDevice, srcImgDevice, cv::ColorConversionCodes::COLOR_BGR2RGBA);

        cv::Mat tst;
        srcImgDevice.download(tst);
        ValidateResolution(nWidth, nHeight);

        ck(cuInit(0));
        int nGpu = 0;
        ck(cuDeviceGetCount(&nGpu));
        if (iGpu < 0 || iGpu >= nGpu)
        {
            std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
            return 1;
        }
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        std::cout << "GPU in use: " << szDeviceName << std::endl;
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));

      
        // Open output file
        std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
        if (!fpOut)
        {
            std::ostringstream err;
            err << "Unable to open output file: " << szOutFilePath << std::endl;
            throw std::invalid_argument(err.str());
        }

        EncodeCuda(nWidth, nHeight, encodeCLIOptions, cuContext, srcImgDevice, fpOut);
        
        fpOut.close();

        std::cout << "Bitstream saved in file " << szOutFilePath << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what();
        return 1;
    }
    return 0;
}
