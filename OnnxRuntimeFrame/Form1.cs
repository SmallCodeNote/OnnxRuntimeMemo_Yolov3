using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using WinFormStringCnvClass;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxRuntimeFrame
{

    public partial class Form1 : Form
    {
        string thisExeDirPath;
        public Form1()
        {
            InitializeComponent();
            thisExeDirPath = Path.GetDirectoryName(Application.ExecutablePath);
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "TEXT|*.txt";
            if (false && ofd.ShowDialog() == DialogResult.OK)
            {
                WinFormStringCnv.setControlFromString(this, File.ReadAllText(ofd.FileName));
            }
            else
            {
                string paramFilename = Path.Combine(thisExeDirPath, "_param.txt");
                if (File.Exists(paramFilename))
                {
                    WinFormStringCnv.setControlFromString(this, File.ReadAllText(paramFilename));
                }
            }
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            string FormContents = WinFormStringCnv.ToString(this);

            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "TEXT|*.txt";

            if (false && sfd.ShowDialog() == DialogResult.OK)
            {
                File.WriteAllText(sfd.FileName, FormContents);
            }
            else
            {
                string paramFilename = Path.Combine(thisExeDirPath, "_param.txt");
                File.WriteAllText(paramFilename, FormContents);
            }
        }

        private void button_OpenOnnxFilePath_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "ONNX|*.onnx";
            if (ofd.ShowDialog() != DialogResult.OK) return;

            textBox_OnnxFilePath.Text = ofd.FileName;
        }

        private void button_OpenImageFilePath_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();

            if (ofd.ShowDialog() != DialogResult.OK) return;

            textBox_ImageFilePath.Text = ofd.FileName;
        }

        private void button_Run_Click(object sender, EventArgs e)
        {
            string onnxFilePath = textBox_OnnxFilePath.Text;
            string imageFilePath = textBox_ImageFilePath.Text;

            using (var session = new InferenceSession(onnxFilePath))
            {
                var inputMeta = session.InputMetadata;
                var inputName = inputMeta.First().Key;
                var inputDims = inputMeta.First().Value.Dimensions;

                using (var bitmap = new Bitmap(Image.FromFile(imageFilePath), inputDims[2], inputDims[3]))
                {
                    var input = new DenseTensor<float>(new[] { 1, 3, inputDims[2], inputDims[3] });
                    for (int y = 0; y < bitmap.Height; y++)
                    {
                        for (int x = 0; x < bitmap.Width; x++)
                        {
                            var color = bitmap.GetPixel(x, y);
                            input[0, 0, y, x] = color.B;
                            input[0, 1, y, x] = color.G;
                            input[0, 2, y, x] = color.R;
                        }
                    }

                    var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(inputName, input) };
                    using (var results = session.Run(inputs))
                    {
                        var output = results.First().AsEnumerable<float>().Select(x => x.ToString()).ToArray();
                        textBox_Result.Text = string.Join(", ", output);
                    }
                }
            }
        }
    }
}
