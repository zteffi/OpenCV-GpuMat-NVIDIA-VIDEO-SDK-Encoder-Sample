# OpenCV GpuMat & NVIDIA VIDEO SDK Encoder Sample
Since `cv::cudacodec::VideoWriter` supports only deprecated `nvcuvnenc` library, It's currently (*OpenCV* 4.3) impossible to encode `cv::cuda::GpuMat` frames without copying them to CPU (`cv::Mat`) through `cudacodec` module. To circumvent this, you can use this sample to encode `cv::cuda::GpuMat` using  *NVIDIA VIDEO SDK* directly. 

## Setup
* Copy `AppTransOpenCV` folder into `Video_Codec_SDK_*.*.*/Samples/AppEncode/`  
* In `Video_Codec_SDK_*.*.*/Samples/CMakeLists.txt` include folder by adding  this line
`add_subdirectory(AppEncode/AppEncOpenCV)`
* Set `OpenCV_DIR` and build sample project with *cmake*
