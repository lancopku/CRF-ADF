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
using System.Diagnostics; 

namespace Program
{
    class MainClass
    {
        static void Main(string[] args)
        {
            Stopwatch timer = new Stopwatch();
            timer.Start();

            Console.WriteLine("CRF-ADF toolkit");
            Console.WriteLine("Copyright(C) Xu Sun <xusun@pku.edu.cn>, All rights reserved.");
            int flag = readCommand(args);
            if (flag == 1)
                return;
            else if (flag == 2)
            {
                Console.WriteLine("command wrong...type 'help' for help on command.");
                return;
            }
            Global.globalCheck();//should check after readCommand()
            directoryCheck();
        
            Global.swLog = new StreamWriter(Global.outDir + Global.fLog);//to record the most detailed runing info
            Global.swResRaw = new StreamWriter(Global.outDir + Global.fResRaw);//to record raw results
            Global.swTune = new StreamWriter(Global.outDir + Global.fTune);//to record tuning info

            Global.swLog.WriteLine("exe command:");
            string cmd = "";
            foreach (string im in args)
                cmd += im + " ";
            Global.swLog.WriteLine(cmd);
            Global.printGlobals();

            if (Global.runMode.Contains("tune"))
            {
                Console.WriteLine("\nstart tune...");
                Global.swLog.WriteLine("\nstart tune...");
                tuneStochasticOptimizer();
            }
            else if (Global.runMode.Contains("train"))//train mode
            {
                Console.WriteLine("\nstart training...");
                Global.swLog.WriteLine("\nstart training...");
                if (Global.runMode.Contains("rich"))
                    richEdge.train();//with rich edge feature: more accurate but slower
                else
                    train();//normal train
            }
            else if (Global.runMode.Contains("test"))//test mode
            {
                Console.WriteLine("\nstart test...");
                Global.swLog.WriteLine("\nstart test...");
                if (Global.runMode.Contains("rich"))
                    richEdge.test();//with rich edge feature: more accurate but slower
                else
                    test();//normal test
            }
            else if (Global.runMode.Contains("cv"))//cross validation mode
            {
                Console.WriteLine("\nstart cross validation...");
                Global.swLog.WriteLine("\nstart cross validation...");
                crossValidation();
            }
            else throw new Exception("error");

            timer.Stop();
            double time = timer.ElapsedMilliseconds / 1000.0;
            Console.WriteLine("\ndone. run time (sec): " + time.ToString());
            Global.swLog.WriteLine("\ndone. run time (sec): " + time.ToString());

            Global.swLog.Close();
            Global.swResRaw.Close();
            Global.swTune.Close();

            resSummarize.summarize();//summarize results
            //Console.ReadLine();
        }

        static double train()
        {
            //load data
            Console.WriteLine("\nreading training & test data...");
            Global.swLog.WriteLine("\nreading training & test data...");
            dataSet X, XX;
            if (Global.runMode.Contains("tune"))//put "tune" related code here because train() could be sub-function of tune()
            {
                dataSet origX = new dataSet(Global.fFeatureTrain, Global.fGoldTrain);
                X = new dataSet();
                XX = new dataSet();
                dataSplit(origX, Global.tuneSplit, X, XX);
            }
            else
            {
                X = new dataSet(Global.fFeatureTrain, Global.fGoldTrain);
                XX = new dataSet(Global.fFeatureTest, Global.fGoldTest);
                dataSizeScale(X);
            }
            Console.WriteLine("done! train/test data sizes: {0}/{1}", X.Count, XX.Count);
            Global.swLog.WriteLine("done! train/test data sizes: {0}/{1}", X.Count, XX.Count);
            double score = 0;

            //start training
            foreach (double r in Global.regList)//train on different r (sigma)
            {
                Global.reg = r;
                Global.swLog.WriteLine("\nr: " + r.ToString());
                Console.WriteLine("\nr: " + r.ToString());
                if (Global.rawResWrite) Global.swResRaw.WriteLine("\n%r: " + r.ToString());
                toolbox tb = new toolbox(X, true);
                score = basicTrain(XX, tb);
                resSummarize.write();//summarize the results & output the summarized results
                
                if (Global.save == 1)
                    tb.Model.save(Global.fModel);//save model as a .txt file
            }
            return score;
        }

        static double test()
        {
            Console.WriteLine("reading test data...");
            Global.swLog.WriteLine("reading test data...");
            dataSet XX = new dataSet(Global.fFeatureTest, Global.fGoldTest);
            Console.WriteLine("Done! test data size: {0}", XX.Count);
            Global.swLog.WriteLine("Done! test data size: {0}", XX.Count);
            //load model & feature files for testing
            toolbox tb = new toolbox(XX, false);

            Stopwatch timer = new Stopwatch();
            timer.Start();

            List<double> scoreList = tb.test(XX, 0);

            timer.Stop();
            double time = timer.ElapsedMilliseconds / 1000.0;

            Global.timeList.Add(time);
            double score = scoreList[0];
            Global.scoreListList.Add(scoreList);

            resSummarize.write();
            return score;
        }

        static void crossValidation()
        {
            //load data
            Console.WriteLine("reading cross validation data...");
            Global.swLog.WriteLine("reading cross validation data...");
            List<dataSet> XList = new List<dataSet>();
            List<dataSet> XXList = new List<dataSet>();
            loadDataForCV(XList, XXList);

            //start cross validation
            foreach (double r in Global.regList)//do CV for each different regularizer r (sigma)
            {
                Global.swLog.WriteLine("\ncross validation. r={0}", r);
                Console.WriteLine("\ncross validation. r={0}", r);
                if (Global.rawResWrite) Global.swResRaw.WriteLine("% cross validation. r={0}", r);
                for (int i = 0; i < Global.nCV; i++)
                {
                    Global.swLog.WriteLine("\n#validation={0}", i + 1);
                    Console.WriteLine("\n#validation={0}", i + 1);
                    if (Global.rawResWrite) Global.swResRaw.WriteLine("% #validation={0}", i + 1);
                    Global.reg = r;
                    dataSet Xi = XList[i];
                    if (Global.runMode.Contains("rich"))
                    {
                        toolboxRich tb = new toolboxRich(Xi);
                        basicTrain(XXList[i], tb);
                    }
                    else
                    {
                        toolbox tb = new toolbox(Xi);
                        basicTrain(XXList[i], tb);
                    }
          
                    resSummarize.write();
                    if (Global.rawResWrite) Global.swResRaw.WriteLine();
                }
                if (Global.rawResWrite) Global.swResRaw.WriteLine();
            }
        }

        static void tuneStochasticOptimizer()//tune parameters rate0 & reg for SGD & ADF training
        {
            if (Global.modelOptimizer.EndsWith("sgd") || Global.modelOptimizer.EndsWith("sgder") || Global.modelOptimizer.EndsWith("adf"))
            {
                //backup
                int origTtlIter = Global.ttlIter;
                List<double> origRegList = new List<double>(Global.regList);
                //change globals
                Global.ttlIter = Global.iterTuneStoch;
                Global.regList = new List<double>();
                Global.regList.Add(1);
                Global.rawResWrite = false;
                //tune rate0 based on reg=1
                //double[] rates = { 0.5, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001 };
                double[] rates = { 0.1, 0.05, 0.01, 0.005 };
                double bestRate = -1, bestScore = 0;
                foreach (double im in rates)
                {
                    Global.rate0 = im;
                    double score = reinitTrain();
                    Global.swTune.WriteLine("reg={0}  rate0={1} --> {2}={3}%", Global.regList[0], im, Global.metric, score.ToString("f2"));
                    Global.swLog.WriteLine("reg={0}  rate0={1} --> {2}={3}%", Global.regList[0], im, Global.metric, score.ToString("f2"));
                    Console.WriteLine("reg={0}  rate0={1} --> {2}={3}%", Global.regList[0], im, Global.metric, score.ToString("f2"));
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestRate = im;
                    }
                }
                Global.rate0 = bestRate;
                //tune reg based on best-rate0
                bestScore = 0;
                double bestReg = -1;
                //double[] regs = { 0.2, 0.5, 1, 2, 5, 10 };
                double[] regs = { 0, 1, 2, 5, 10 };
                Global.swTune.WriteLine();
                foreach (double im in regs)
                {
                    Global.regList.Clear();
                    Global.regList.Add(im);
                    double score = reinitTrain();
                    Global.swTune.WriteLine("reg={0}  rate0={1} --> {2}={3}%", Global.regList[0], Global.rate0, Global.metric, score.ToString("f2"));
                    Global.swLog.WriteLine("reg={0}  rate0={1} --> {2}={3}%", Global.regList[0], Global.rate0, Global.metric, score.ToString("f2"));
                    Console.WriteLine("reg={0}  rate0={1} --> {2}={3}%", Global.regList[0], Global.rate0, Global.metric, score.ToString("f2"));
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestReg = im;
                    }
                }
                Global.reg = bestReg;
                Global.swTune.WriteLine("\nconclusion: best-rate0={0}, best-reg={1}", Global.rate0, Global.reg);
                //recover setting
                Global.ttlIter = origTtlIter;
                Global.regList = new List<double>(origRegList);
                Global.rawResWrite = true;

                Console.WriteLine("done");
            }
            else
            {
                Console.WriteLine("no need tuning for non-stochastic optimizer! done.");
                Global.swLog.WriteLine("no need tuning for non-stochastic optimizer! done.");
            }
        }

        static double reinitTrain()
        {
            Global.reinitGlobal();
            double score = 0;
            if (Global.runMode.Contains("rich"))
                score = richEdge.train();
            else
                score = train();
            return score;
        }

        //this function can be called by train(), cv(), & richEdge.train()
        public static double basicTrain(dataSet XTest, toolbox tb)
        {
            Global.reinitGlobal();
            double score = 0;

            if (Global.modelOptimizer.EndsWith("bfgs"))
            {
                Global.tb = tb;
                Global.XX = XTest;

                tb.train();
                score = Global.scoreListList[Global.scoreListList.Count - 1][0];
            }
            else
            {
                for (int i = 0; i < Global.ttlIter; i++)
                {
                    Global.glbIter++;
                    Stopwatch timer = new Stopwatch();
                    timer.Start();

                    double err = tb.train();

                    timer.Stop();
                    double time = timer.ElapsedMilliseconds/1000.0;

                    Global.timeList.Add(time);
                    Global.errList.Add(err);
                    Global.diffList.Add(Global.diff);

                    List<double> scoreList = tb.test(XTest, i);
                    score = scoreList[0];
                    Global.scoreListList.Add(scoreList);

                    Global.swLog.WriteLine("iter{0}  diff={1}  train-time(sec)={2}  {3}={4}%", Global.glbIter, Global.diff.ToString("e2"), time.ToString("f2"), Global.metric, score.ToString("f2"));
                    Global.swLog.WriteLine("------------------------------------------------");
                    Global.swLog.Flush();
                    Console.WriteLine("iter{0}  diff={1}  train-time(sec)={2}  {3}={4}%", Global.glbIter, Global.diff.ToString("e2"), time.ToString("f2"), Global.metric, score.ToString("f2"));

                    //if (Global.diff < Global.convergeTol)
                        //break;
                }
            }
            return score;
        }

        public static void dataSizeScale(dataSet X)
        {
            dataSet XX = new dataSet();
            XX.setDataInfo(X);
            foreach (dataSeq im in X)
                XX.Add(im);
            X.Clear();

            int n = (int)(XX.Count * Global.trainSizeScale);
            for (int i = 0; i < n; i++)
            {
                int j = i;
                if (j > XX.Count - 1)
                    j %= XX.Count - 1;
                X.Add(XX[j]);
            }
            X.setDataInfo(XX);
        }

        public static void dataSplit(dataSet X, double v1, double v2, dataSet X1, dataSet X2)
        {
            if (v2 < v1)
                throw new Exception("error");
            X1.Clear();
            X2.Clear();
            X1.setDataInfo(X);
            X2.setDataInfo(X);
            int n1 = (int)(X.Count * v1);
            int n2 = (int)(X.Count * v2);
            for (int i = 0; i < X.Count; i++)
            {
                if (i >= n1 && i < n2)
                    X1.Add(X[i]);
                else
                    X2.Add(X[i]);
            }
        }

        public static void dataSplit(dataSet X, double v, dataSet X1, dataSet X2)
        {
            X1.Clear();
            X2.Clear();
            X1.setDataInfo(X);
            X2.setDataInfo(X);
            int n = (int)(X.Count * v);
            for (int i = 0; i < X.Count; i++)
            {
                if (i < n)
                    X1.Add(X[i]);
                else
                    X2.Add(X[i]);
            }
        }

        public static void loadDataForCV(List<dataSet> XList, List<dataSet> XXList)
        {
            XList.Clear();
            XXList.Clear();
            //load train data only: CV is based only on training data
            dataSet X = new dataSet(Global.fFeatureTrain, Global.fGoldTrain);
            double step = 1.0 / Global.nCV;
            for (double i = 0; i < 1; i += step)
            {
                dataSet Xi = new dataSet();
                dataSet XRest_i = new dataSet();
                dataSplit(X, i, i + step, Xi, XRest_i);
                XList.Add(XRest_i);
                XXList.Add(Xi);
            }

            Console.WriteLine("Done! cross-validation train/test data sizes (cv_1, ..., cv_n): ");
            Global.swLog.WriteLine("Done! cross-validation train/test data sizes (cv_1, ..., cv_n): ");
            for (int i = 0; i < Global.nCV; i++)
            {
                Global.swLog.WriteLine("{0}/{1}, ", XList[i].Count, XXList[i].Count);
            }
        }

        static void directoryCheck()//check & set directory environment
        {
            if (!Directory.Exists(Global.modelDir))
                Directory.CreateDirectory(Global.modelDir);
            Global.outDir = Directory.GetCurrentDirectory() + "/" + Global.outFolder + "/";
            if (!Directory.Exists(Global.outDir))
                Directory.CreateDirectory(Global.outDir);
            fileTool.removeFile(Global.outDir);
        }

        //should only read command here, should do the command validity check in globalCheck()
        static int readCommand(string[] args)
        {
            foreach (string arg in args)
            {
                if (arg == "help")
                {
                    helpCommand();
                    return 1;
                }
                string[] ary = arg.Split(Global.colonAry, StringSplitOptions.RemoveEmptyEntries);
                if (ary.Length != 2)
                    return 2;
                string opt = ary[0], val = ary[1];

                switch (opt)
                {
                    case "m":
                        Global.runMode = val;
                        break;
                    case "mo":
                        Global.modelOptimizer = val;
                        break;
                    case "a":
                        Global.rate0 = double.Parse(val);
                        break;
                    case "r":
                        Global.regList.Clear();
                        string[] regAry = val.Split(Global.commaAry, StringSplitOptions.RemoveEmptyEntries);
                        foreach (string im in regAry)
                        {
                            double reg = double.Parse(im);
                            Global.regList.Add(reg);
                        }
                        break;
                    case "d":
                        Global.random = int.Parse(val);
                        break;
                    case "e":
                        Global.evalMetric = val;
                        break;
                    //case "t":
                    //   Global.taskBasedChunkInfo = val;
                    // break;
                    case "ss":
                        Global.trainSizeScale = double.Parse(val);
                        break;
                    case "i":
                        Global.ttlIter = int.Parse(val);
                        break;
                    case "n":
                        Global.nUpdate = int.Parse(val);
                        break;
                    case "s":
                        if (val == "1")
                            Global.save = 1;
                        else
                            Global.save = 0;
                        break;
                    case "of":
                        Global.outFolder = val;
                        break;
                    case "mb":
                        Global.miniBatch = int.Parse(val);
                        break;
                    case "up":
                        Global.upper = double.Parse(val);
                        break;
                    case "lw":
                        Global.lower = double.Parse(val);
                        break;
                    default:
                        return 2;
                }
            }
            return 0;//success
        }

        static void helpCommand()
        {
            Console.WriteLine("'option1:value1  option2:value2 ...' for setting values to options.");
            Console.WriteLine("'m' for runMode. Default: {0}", Global.runMode);
            Console.WriteLine("'mo' for modelOptimizer. Default: {0}", Global.modelOptimizer);
            Console.WriteLine("'a' for rate0. Default: {0}", Global.rate0);
            Console.WriteLine("'r' for regList. E.g., 'r:1,2,3'. Default: {0}", Global.regList[0]);
            Console.WriteLine("'d' for random. Default: {0}", Global.random);
            Console.WriteLine("'e' for evalMetric. Default: {0}", Global.evalMetric);
            //Console.WriteLine("'t' for taskBasedChunkInfo. Default: {0}", Global.taskBasedChunkInfo);
            Console.WriteLine("'ss' for trainSizeScale. Default: {0}", Global.trainSizeScale);
            Console.WriteLine("'i' for ttlIter. Default: {0}", Global.ttlIter);
            Console.WriteLine("'n' for nUpdate. Default: {0}", Global.nUpdate);
            Console.WriteLine("'s' for save. Default: {0}", Global.save);
            Console.WriteLine("'of' for outFolder. Default: {0}", Global.outFolder);
            Console.WriteLine("'mb' for miniBatch. Default: {0}", Global.miniBatch);
            Console.WriteLine("'up' for ADF decay rate upper-bound. Default: {0}", Global.upper);
            Console.WriteLine("'lw' for ADF decay rate lower-bound. Default: {0}", Global.lower);
        }
    }
}
