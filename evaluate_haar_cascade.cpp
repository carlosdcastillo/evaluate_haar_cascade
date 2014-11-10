
#include <math.h>
#include <sys/types.h>
#include <assert.h>
#include <string.h>

#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//typedef uint16_t char16_t;
#include "mex.h"
#include "matrix.h"
#include "string.h"


/*Compile using the following command:
 
/Applications/MATLAB_R2011a.app/bin/mex -v -largeArrayDims CXXFLAGS='-I/opt/local/include/opencv -I/opt/local/include -O3 -pipe -fomit-frame-pointer -Wall -Wconversion -funroll-loops -msse -msse2 -mfpmath=sse -funit-at-a-time' LDFLAGS='-Wl,-twolevel_namespace -undefined error -arch x86_64 -Wl,-syslibroot,/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk/ -mmacosx-version-min=10.5 -bundle -Wl,-exported_symbols_list,/Applications/MATLAB_R2011a.app/extern/lib/maci64/mexFunction.map -L/opt/local/lib' CXXLIBS='/Users/carlos/opencv-2.4.8/build/3rdparty/lib/libIlmImf.a /Users/carlos/opencv-2.4.8/build/3rdparty/lib/liblibjasper.a /Users/carlos/opencv-2.4.8/build/3rdparty/lib/liblibjpeg.a /Users/carlos/opencv-2.4.8/build/3rdparty/lib/liblibpng.a /Users/carlos/opencv-2.4.8/build/3rdparty/lib/liblibtiff.a /Users/carlos/opencv-2.4.8/build/3rdparty/lib/libzlib.a  /Users/carlos/opencv-2.4.8/build/lib/libopencv_core.a /Users/carlos/opencv-2.4.8/build/lib/libopencv_haartraining_engine.a /Users/carlos/opencv-2.4.8/build/lib/libopencv_imgproc.a   /Users/carlos/opencv-2.4.8/build/lib/libopencv_objdetect.a  -L/Applications/MATLAB_R2011a.app/bin/maci64 -lmx -lmex -lmat -lc++' evaluate_haar_cascade.cpp

*/

#define GET(dataptr,MX,MY,MZ,i,j,k) dataptr[(i) + (MX) * ((j) + (MY) * (k))]

std::vector<cv::Rect> face_detect(cv::Mat& im, cv::CascadeClassifier& cascade)
{
    cv::Mat gray;
	if (im.channels() == 3){
        im.convertTo(gray,CV_8UC1);
	}else{
		im.copyTo(gray);
	}
    std::vector<cv::Rect> results;
    cascade.detectMultiScale(gray,results,1.1,3,CV_HAAR_DO_CANNY_PRUNING,cvSize(20,20));
	return results;
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{

    unsigned char* image = (unsigned char*)mxGetPr(prhs[0]);
    const mwSize *imsz = mxGetDimensions(prhs[0]);
    char haar_cascade_filename[255];
    mxGetString(prhs[1],haar_cascade_filename,255);
    cv::Mat img(imsz[0],imsz[1], CV_8UC3, cv::Scalar(0,0,255));

    for (mwSize i = 0; i < imsz[0]; i++){
        for (mwSize j = 0; j < imsz[1]; j++){
            for (int k = 0; k < 3; k++){

                img.at<cv::Vec3b>(i,j)[2-k] = GET(image, imsz[0], imsz[1],
                                                imsz[2], i, j, k);


            }
        }
    }

    std::string haarfn = std::string(haar_cascade_filename);
    cv::CascadeClassifier cascade(haarfn);

    std::vector<cv::Rect> detections = face_detect(img, cascade);

    plhs[0] = mxCreateDoubleMatrix(detections.size(),4,mxREAL);
    double* res = mxGetPr(plhs[0]);
    
    int pos = 0;
    for(std::vector<cv::Rect>::iterator it = detections.begin(); it !=
        detections.end(); it++){
        cv::Rect rect = *it;
        GET(res,detections.size(),4,1,pos,0,0) = rect.x + 1;
        GET(res,detections.size(),4,1,pos,1,0) = rect.x + rect.width + 1;
        GET(res,detections.size(),4,1,pos,2,0) = rect.y + 1;
        GET(res,detections.size(),4,1,pos,3,0) =  rect.y + rect.height + 1;
        pos++;
    }

}

