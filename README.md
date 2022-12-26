# OpenCV GpuMat & NVIDIA VIDEO SDK Encoder Sample
Since `cv::cudacodec::VideoWriter` supports only deprecated `nvcuvnenc` library, It's currently (*OpenCV* 4.3) impossible to encode `cv::cuda::GpuMat` frames without copying them to CPU (`cv::Mat`) through `cudacodec` module. To circumvent this, you can use this sample to encode `cv::cuda::GpuMat` using  *NVIDIA VIDEO SDK* directly. 

## Setup
* Copy `AppEncOpenCV` folder into `Video_Codec_SDK_*.*.*/Samples/AppEncode/`  
* In `Video_Codec_SDK_*.*.*/Samples/CMakeLists.txt` include folder by adding  this line
`add_subdirectory(AppEncode/AppEncOpenCV)`
* Set `OpenCV_DIR` and build sample project with *cmake*

# Usage
`./AppEncOpenCV -i path_to_image.jpg -o video.h264`

Since the output is just the raw encoded audio, we need to mux it into some video container to view it in most media players, so just run

`ffmpeg -i video.h264 video.mp4`

Sample will produce 15 second video with input image as it's frames. Alternatively you can just check `EncodeGpuMat` function and use it in your application. The most important part is using `NV_ENC_BUFFER_FORMAT_ABGR` when initializing `NvEncoderCuda`. Then you can use `NvEncoderCuda::CopyToDeviceFrame` with `cv::GpuMat::data`  as `void* pSrcFrame`  argument.

```c++

cv::cuda::GpuMat srcIn;
NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_ABGR;
std::unique_ptr<NvEncoderCuda> pEnc(new NvEncoderCuda(cuContext, nWidth, nHeight, eFormat));

...

const NvEncInputFrame* encoderInputFrame = pEnc->GetNextInputFrame();

NvEncoderCuda::CopyToDeviceFrame(
	cuContext, 
	srcIn.data, 
	0, 
	(CUdeviceptr)encoderInputFrame->inputPtr,
	(int)encoderInputFrame->pitch,
	pEnc->GetEncodeWidth(),
	pEnc->GetEncodeHeight(),
	CU_MEMORYTYPE_HOST,
	encoderInputFrame->bufferFormat,
	encoderInputFrame->chromaOffsets,
	encoderInputFrame->numChromaPlanes);
	
pEnc->EncodeFrame(vPacket);

```