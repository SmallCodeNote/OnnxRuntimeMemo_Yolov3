using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxYolov3Loader
{
    /// <summary>
    /// yolov3 model source
    /// https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/yolov3
    /// </summary>

    static class OnnxYolov3
    {
        static string[] cocoLabels = new string[] { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear","hair drier", "toothbrush" };

        static public string RunSession(string onnxFilePath, string imageFilePath, int inputWidth = 416, int inputHeight = 416, bool ImShow = false)
        {
            Mat imgSrc = Cv2.ImRead(imageFilePath, ImreadModes.Color);
            string dstString = RunSessionAndDrawMat(onnxFilePath, imgSrc, inputWidth, inputHeight, ImShow);
            imgSrc.Dispose();

            return dstString;
        }

        static public string RunSessionAndDrawMat(string onnxFilePath, Mat imgSrc, int inputWidth = 416, int inputHeight = 416, bool ImShow = false)
        {
            using (var session = new InferenceSession(onnxFilePath))
            {
                return RunSessionAndDrawMat(session, imgSrc, inputWidth, inputHeight, ImShow);
            }
        }

        static public string RunSessionAndDrawMat(InferenceSession session, Mat imgSrc, int inputWidth = 416, int inputHeight = 416, bool ImShow = false)
        {
            var input = getDenseTensorFromMat(imgSrc, inputWidth, inputHeight);

            var inputShape = new DenseTensor<float>(new[] { 1, 2 });
            inputShape[0, 0] = imgSrc.Height;
            inputShape[0, 1] = imgSrc.Width;

            var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("input_1", input),
                        NamedOnnxValue.CreateFromTensor("image_shape", inputShape)
                    };

            List<string> LineOutput = new List<string>();

            using (var results = session.Run(inputs))
            {
                Tensor<float> boxes = results[0].AsTensor<float>();
                var scores = results[1].AsTensor<float>();
                var indices = results[2].AsTensor<int>();

                // Process results
                int indicesLength = indices.Dimensions[0];
                for (int i = 0; i < indicesLength; i++)
                {
                    var score = scores[indices[i, 0]];
                    var label = cocoLabels[indices[i, 1]];
                    int boxIndex = indices[i, 2];

                    var p1_y = boxes[0, boxIndex, 0];
                    var p1_x = boxes[0, boxIndex, 1];
                    var p2_y = boxes[0, boxIndex, 2];
                    var p2_x = boxes[0, boxIndex, 3];

                    DrawRectangleAndLabel(imgSrc, p1_x, p1_y, p2_x, p2_y, label);
                    LineOutput.Add(label + "\t" + score.ToString("g4") + "\t" + p1_x.ToString("0.0") + "\t" + p1_y.ToString("0.0") + "\t" + p2_x.ToString("0.0") + "\t" + p2_y.ToString("0.0"));
                }
            }

            if (ImShow)
            {
                Cv2.ImShow("Image", imgSrc);
                Cv2.WaitKey(0);
                Cv2.DestroyAllWindows();
            }

            return string.Join("\r\n", LineOutput.ToArray());
        }

        static private DenseTensor<float> getDenseTensorFromMat(Mat src, int tensorWidth, int tensorHeight)
        {
            var dstTensor = new DenseTensor<float>(new[] { 1, 3, tensorWidth, tensorHeight });

            Size newSize = new Size(tensorWidth, tensorHeight);
            Mat dst = new Mat();
            Cv2.Resize(src, dst, newSize); 

            for (int y = 0; y < tensorHeight; y++)
            {
                for (int x = 0; x < tensorWidth; x++)
                {
                    Vec3b color = dst.At<Vec3b>(y, x);
                    dstTensor[0, 0, y, x] = ((float)color.Item2) / 255f;
                    dstTensor[0, 1, y, x] = ((float)color.Item1) / 255f;
                    dstTensor[0, 2, y, x] = ((float)color.Item0) / 255f;
                }
            }

            dst.Dispose();
            return dstTensor;
        }

        static private void DrawRectangleAndLabel(Mat src, float p1_x, float p1_y, float p2_x, float p2_y, string label)
        {
            OpenCvSharp.Point p1 = new OpenCvSharp.Point(p1_x, p1_y);
            OpenCvSharp.Point p2 = new OpenCvSharp.Point(p2_x, p2_y);

            Cv2.Rectangle(src, p1, p2, Scalar.Red, thickness: 2);
            Cv2.PutText(src, label, p1, HersheyFonts.HersheySimplex, 1.0, Scalar.LightSkyBlue, thickness: 2);
        }
    }
}
