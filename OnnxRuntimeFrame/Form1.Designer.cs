﻿namespace OnnxRuntimeFrame
{
    partial class Form1
    {
        /// <summary>
        /// 必要なデザイナー変数です。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 使用中のリソースをすべてクリーンアップします。
        /// </summary>
        /// <param name="disposing">マネージド リソースを破棄する場合は true を指定し、その他の場合は false を指定します。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows フォーム デザイナーで生成されたコード

        /// <summary>
        /// デザイナー サポートに必要なメソッドです。このメソッドの内容を
        /// コード エディターで変更しないでください。
        /// </summary>
        private void InitializeComponent()
        {
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.button_OpenOnnxFilePath = new System.Windows.Forms.Button();
            this.textBox_OnnxFilePath = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.button_Run = new System.Windows.Forms.Button();
            this.button_OpenImageFilePath = new System.Windows.Forms.Button();
            this.textBox_ImageFilePath = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.textBox_Result = new System.Windows.Forms.TextBox();
            this.SuspendLayout();
            // 
            // statusStrip1
            // 
            this.statusStrip1.Location = new System.Drawing.Point(0, 576);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(660, 22);
            this.statusStrip1.TabIndex = 0;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // button_OpenOnnxFilePath
            // 
            this.button_OpenOnnxFilePath.Location = new System.Drawing.Point(531, 30);
            this.button_OpenOnnxFilePath.Name = "button_OpenOnnxFilePath";
            this.button_OpenOnnxFilePath.Size = new System.Drawing.Size(22, 23);
            this.button_OpenOnnxFilePath.TabIndex = 1;
            this.button_OpenOnnxFilePath.Text = "...";
            this.button_OpenOnnxFilePath.UseVisualStyleBackColor = true;
            this.button_OpenOnnxFilePath.Click += new System.EventHandler(this.button_OpenOnnxFilePath_Click);
            // 
            // textBox_OnnxFilePath
            // 
            this.textBox_OnnxFilePath.Location = new System.Drawing.Point(12, 30);
            this.textBox_OnnxFilePath.Name = "textBox_OnnxFilePath";
            this.textBox_OnnxFilePath.Size = new System.Drawing.Size(513, 19);
            this.textBox_OnnxFilePath.TabIndex = 2;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(19, 14);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(73, 12);
            this.label1.TabIndex = 3;
            this.label1.Text = "OnnxFilePath";
            // 
            // button_Run
            // 
            this.button_Run.Location = new System.Drawing.Point(22, 123);
            this.button_Run.Name = "button_Run";
            this.button_Run.Size = new System.Drawing.Size(69, 25);
            this.button_Run.TabIndex = 4;
            this.button_Run.Text = "Run";
            this.button_Run.UseVisualStyleBackColor = true;
            this.button_Run.Click += new System.EventHandler(this.button_Run_Click);
            // 
            // button_OpenImageFilePath
            // 
            this.button_OpenImageFilePath.Location = new System.Drawing.Point(531, 85);
            this.button_OpenImageFilePath.Name = "button_OpenImageFilePath";
            this.button_OpenImageFilePath.Size = new System.Drawing.Size(22, 23);
            this.button_OpenImageFilePath.TabIndex = 1;
            this.button_OpenImageFilePath.Text = "...";
            this.button_OpenImageFilePath.UseVisualStyleBackColor = true;
            this.button_OpenImageFilePath.Click += new System.EventHandler(this.button_OpenImageFilePath_Click);
            // 
            // textBox_ImageFilePath
            // 
            this.textBox_ImageFilePath.Location = new System.Drawing.Point(12, 85);
            this.textBox_ImageFilePath.Name = "textBox_ImageFilePath";
            this.textBox_ImageFilePath.Size = new System.Drawing.Size(513, 19);
            this.textBox_ImageFilePath.TabIndex = 2;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(19, 69);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(77, 12);
            this.label2.TabIndex = 3;
            this.label2.Text = "ImageFilePath";
            // 
            // textBox_Result
            // 
            this.textBox_Result.Location = new System.Drawing.Point(27, 191);
            this.textBox_Result.Multiline = true;
            this.textBox_Result.Name = "textBox_Result";
            this.textBox_Result.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.textBox_Result.Size = new System.Drawing.Size(538, 355);
            this.textBox_Result.TabIndex = 5;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(660, 598);
            this.Controls.Add(this.textBox_Result);
            this.Controls.Add(this.button_Run);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.textBox_ImageFilePath);
            this.Controls.Add(this.button_OpenImageFilePath);
            this.Controls.Add(this.textBox_OnnxFilePath);
            this.Controls.Add(this.button_OpenOnnxFilePath);
            this.Controls.Add(this.statusStrip1);
            this.Name = "Form1";
            this.Text = "OnnxRuntimeFrame";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form1_FormClosing);
            this.Load += new System.EventHandler(this.Form1_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.Button button_OpenOnnxFilePath;
        private System.Windows.Forms.TextBox textBox_OnnxFilePath;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button button_Run;
        private System.Windows.Forms.Button button_OpenImageFilePath;
        private System.Windows.Forms.TextBox textBox_ImageFilePath;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox textBox_Result;
    }
}

