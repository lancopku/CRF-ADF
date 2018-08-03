//
//  CRF-ADF Toolkit v1.0
//
//  Copyright(C) Xu Sun <xusun@pku.edu.cn> http://klcl.pku.edu.cn/member/sunxu/index.htm
//

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Collections;
using System.IO.Compression;
using System.Threading;

namespace Program
{
    class Global
    {
        //default values
        public static string runMode = "train";//train (normal training), train.rich (training with rich edge features), test, tune£¬ tune.rich, cv (cross validation), cv.rich
        public static string modelOptimizer = "crf.adf";//crf.sgd/sgder/adf/bfgs
        public static double rate0 = 0.05;//init value of decay rate in SGD and ADF training
        static double[] regs = { 1,3,5};
        public static int random = 0;//0 for 0-initialization of model weights, 1 for random init of model weights
        public static string evalMetric = "tok.acc";//tok.acc (token accuracy), str.acc (string accuracy), f1 (F1-score)
        public static double trainSizeScale = 1;//for scaling the size of training data
        public static int ttlIter = 100;//# of training iterations
        public static int nUpdate = 10;//for ADF training
        public static string outFolder = "out";
        public static int save = 1;//save model file
        public static bool rawResWrite = true;
        public static int miniBatch = 1;//mini-batch in stochastic training
        public static int nCV = 4;//automatic #-fold cross validation
        public static List<List<dataSeqTest>> threadXX;
        public static int nThread = 10;
        public static double edgeReduce = 0.4;
        public static bool useTraditionalEdge = true;
        //ADF training
        public static double upper = 0.995;//was tuned for nUpdate = 10
        public static double lower = 0.6;//was tuned for nUpdate = 10

        //general
        public const double tuneSplit = 0.8;//size of data split for tuning
        public static bool debug = false;//some debug code will run in debug mode
        //SGD training
        public const double decayFactor = 0.94;//decay factor in SGD training
        public const int scalarResetStep = 1000;
        //LBFGS training
        public const int mBFGS = 10;//history of 10 iterations of gradients to estimate Hessian info
        public const bool wolfe = false;//for convex & non-convex objective function
        //tuning
        public const int iterTuneStoch = 30;//default 30

        //global variables
        public static baseHashMap<int, string> chunkTagMap = new baseHashMap<int, string>();
        public static string metric;
        public static List<double> regList = new List<double>(regs);
        public static double ttlScore = 0;
        public static int interval;
        public static toolbox tb;
        public static dataSet XX;
        public static List<double> decayList;
        public static string outDir = "";
        public static List<List<double>> scoreListList = new List<List<double>>();
        public static List<double> timeList = new List<double>();
        public static List<double> errList = new List<double>();
        public static List<double> diffList = new List<double>();
        public static double reg = 1;
        public static int glbIter = 0;
        public static double diff = 1e100;//relative difference from the previous object value, for convergence test
        public static int countWithIter = 0;
        public static StreamWriter swTune;
        public static StreamWriter swLog;
        public static StreamWriter swResRaw;
        public static StreamWriter swOutput;
        public const string fTune = "tune.txt";
        public const string fLog = "trainLog.txt";
        public const string fResSum = "summarizeResult.txt";
        public const string fResRaw = "rawResult.txt";
        public const string fFeatureTrain = "ftrain.txt";
        public const string fGoldTrain = "gtrain.txt";
        public const string fFeatureTest = "ftest.txt";
        public const string fGoldTest = "gtest.txt";
        public const string fOutput = "outputTag.txt";
        public const string fModel = "model/model.txt";
        public const string modelDir = "model/";
        public static char[] lineEndAry = { '\n' };
        public static string[] biLineEndAry = { "\n\n" };
        public static string[] triLineEndAry = { "\n\n\n" };
        public static char[] barAry = { '-' };
        public static char[] dotAry = { '.'};
        public static char[] underLineAry = { '_' };
        public static char[] commaAry = { ',' };
        public static char[] tabAry = { '\t' };
        public static char[] vertiBarAry = { '|' };
        public static char[] colonAry = { ':' };
        public static char[] blankAry = { ' ' };
        public static char[] starAry = { '*' };
        public static char[] slashAry = { '/' };
 
        public static void reinitGlobal()
        {
            diff = 1e100;
            countWithIter = 0;
            glbIter = 0;
        }

        public static void globalCheck()
        {
            if (runMode.Contains("test"))
                ttlIter = 1;

            if (evalMetric == "f1")
                getChunkTagMap();

            if (evalMetric == "f1")
                metric = "f-score";
            else if (evalMetric == "tok.acc")
                metric = "token-accuracy";
            else if (evalMetric == "str.acc")
                metric = "string-accuracy";
            else throw new Exception("error");

            if (Global.rate0 <= 0)
                throw new Exception("error");
            if (Global.trainSizeScale <= 0)
                throw new Exception("error");
            if (Global.ttlIter <= 0)
                throw new Exception("error");
            if (Global.nUpdate <= 0)
                throw new Exception("error");
            if (Global.miniBatch <= 0)
                throw new Exception("error");
            foreach (double reg in regList)
            {
                if (reg < 0)
                    throw new Exception("error");
            }
        }

        public static void printGlobals()
        {
            swLog.WriteLine("mode: {0}", Global.runMode);
            swLog.WriteLine("modelOptimizer: {0}", Global.modelOptimizer);
            swLog.WriteLine("rate0: {0}", Global.rate0);
            swLog.WriteLine("regs: {0}", Global.regList[0]);
            swLog.WriteLine("random: {0}", Global.random);
            swLog.WriteLine("evalMetric: {0}", Global.evalMetric);
            //swLog.WriteLine("taskBasedChunkInfo: {0}", Global.taskBasedChunkInfo);
            swLog.WriteLine("trainSizeScale: {0}", Global.trainSizeScale);
            swLog.WriteLine("ttlIter: {0}", Global.ttlIter);
            swLog.WriteLine("nUpdate: {0}", Global.nUpdate);
            //swLog.WriteLine("tune: {0}", Global.tune);
            swLog.WriteLine("outFolder: {0}", Global.outFolder);
            swLog.WriteLine("miniBatch: {0}", Global.miniBatch);
            swLog.WriteLine("upper: {0}", Global.upper);
            swLog.WriteLine("lower: {0}", Global.lower);
            swLog.Flush();
        }

        //the system must know the B (begin-chunk), I (in-chunk), O (out-chunk) information for computing f-score
        //since such BIO information is task-dependent, it should be explicitly coded here
        static void getChunkTagMap()
        {
            chunkTagMap.Clear();

            //read the labelMap.txt for chunk tag information
            StreamReader sr = new StreamReader("tagIndex.txt");
            string a = sr.ReadToEnd();
            a = a.Replace("\r", "");
            string[] ary = a.Split(Global.lineEndAry, StringSplitOptions.RemoveEmptyEntries);
            foreach (string im in ary)
            {
                string[] imAry = im.Split(Global.blankAry, StringSplitOptions.RemoveEmptyEntries);
                int index = int.Parse(imAry[1]);
                string[] tagAry = imAry[0].Split(Global.starAry, StringSplitOptions.RemoveEmptyEntries);
                string tag = tagAry[tagAry.Length - 1];//the last tag is the current tag
                //merge I-tag/O-tag: no need to use diversified I-tag/O-tag in computing F-score
                if (tag.StartsWith("I"))
                    tag = "I";
                if (tag.StartsWith("O"))
                    tag = "O";
                chunkTagMap[index] = tag;
            }

            sr.Close();
        }

    }

}
