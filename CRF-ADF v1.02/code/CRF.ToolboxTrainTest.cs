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
using System.Diagnostics;
using System.Threading.Tasks;
using System.Threading;

namespace Program
{
    class toolbox
    {
        protected dataSet _X;
        protected model _model;
        protected optimizer _optim;
        protected inference _inf;
        protected featureGenerator _fGene;
        protected gradient _grad;

        public toolbox()
        {
        }

        public toolbox(dataSet X, bool train = true)
        {
            if (train)//to train
            {
                _X = X;
                _fGene = new featureGenerator(X);
                _model = new model(X, _fGene);
                _inf = new inference(this);
                _grad = new gradient(this);
                initOptimizer();
            }
            else//to test
            {
                _X = X;
                _model = new model(Global.fModel);
                _fGene = new featureGenerator(X);
                _inf = new inference(this);
                _grad = new gradient(this);
            }
        }

        public void initOptimizer()
        {
            //_optimType = Global.optimizer;
            if (Global.modelOptimizer.StartsWith("crf"))
            {
                if (Global.modelOptimizer.EndsWith("sgd") || Global.modelOptimizer.EndsWith("sgder") || Global.modelOptimizer.EndsWith("adf"))
                {
                    _optim = new optimStochastic(this);
                }
                else if (Global.modelOptimizer.EndsWith("bfgs"))
                {
                    _optim = new optimLBFGS(this, _model.W, Global.mBFGS, 0, Global.ttlIter);
                }
                else throw new Exception("error");
            }
            else throw new Exception("error");
        }

        public double train()
        {
            //Start training
            double err = _optim.optimize();

            return err;
        }

        public List<double> test(dataSet X, double iter)
        {
            string outfile = Global.outDir + Global.fOutput;
            Global.swOutput = new StreamWriter(outfile);
            List<double> scoreList;
            if (Global.evalMetric == "tok.acc")
                scoreList = decode_tokAcc(X, _model, iter);
            else if (Global.evalMetric == "str.acc")
                scoreList = decode_strAcc(X, _model, iter);
            else if (Global.evalMetric == "f1")
                scoreList = decode_fscore(X, _model, iter);
            else throw new Exception("error");
            Global.swOutput.Close();

            return scoreList;
        }

        //token accuracy
        public List<double> decode_tokAcc(dataSet X, model m, double iter)
        {
            int nTag = m.NTag;
            int[] tmpAry = new int[nTag];
            List<int> corrOutput = new List<int>(tmpAry);
            List<int> gold = new List<int>(tmpAry);
            List<int> output = new List<int>(tmpAry);

            //multi thread
            List<dataSeqTest> X2 = new List<dataSeqTest>();
            multiThreading(X, X2);

            foreach (dataSeqTest x in X2)
            {
                List<int> outTags = x._yOutput;
                List<int> goldTags = x._x.getTags();

                //output tag results
                if (Global.swOutput != null)
                {
                    for (int i = 0; i < outTags.Count; i++)
                    {
                        Global.swOutput.Write(outTags[i].ToString()+",");
                    }
                    Global.swOutput.WriteLine();
                }

                //count
                for (int i = 0; i < outTags.Count; i++)
                {
                    gold[goldTags[i]]++;
                    output[outTags[i]]++;

                    if (outTags[i] == goldTags[i])
                        corrOutput[outTags[i]]++;
                }
            }
            
            Global.swLog.WriteLine("% tag-type  #gold  #output  #correct-output  token-precision  token-recall  token-f-score");
            double prec, rec;
            int sumGold = 0, sumOutput = 0, sumCorrOutput = 0;
            for (int i = 0; i < nTag; i++)
            {
                sumCorrOutput += corrOutput[i];
                sumGold += gold[i];
                sumOutput += output[i];
                if (gold[i] == 0)
                    rec = 0;
                else
                    rec = ((double)corrOutput[i]) * 100.0 / (double)gold[i];
                if (output[i] == 0)
                    prec = 0;
                else
                    prec = ((double)corrOutput[i]) * 100.0 / (double)output[i];

                Global.swLog.WriteLine("% {0}:  {1}  {2}  {3}  {4}  {5}  {6}", i, gold[i], output[i], corrOutput[i], prec.ToString("f2"), rec.ToString("f2"), (2 * prec * rec / (prec + rec)).ToString("f2"));
            }
            if (sumGold == 0)
                rec = 0;
            else
                rec = ((double)sumCorrOutput) * 100.0 / (double)sumGold;
            if (sumOutput == 0)
                prec = 0;
            else
                prec = ((double)sumCorrOutput) * 100.0 / (double)sumOutput;

            double fscore;
            if (prec == 0 && rec == 0)
                fscore = 0;
            else
                fscore = 2 * prec * rec / (prec + rec);//this token-based overall-f-score is also the token-based-accuracy

            Global.swLog.WriteLine("% overall-tags:  {0}  {1}  {2}  {3}  {4}  {5}", sumGold, sumOutput, sumCorrOutput, prec.ToString("f2"), rec.ToString("f2"), fscore.ToString("f2"));
            Global.swLog.Flush();
            List<double> scoreList = new List<double>();
            scoreList.Add(fscore);
            return scoreList;
        }

        //string accuracy
        public List<double> decode_strAcc(dataSet X, model m, double iter)
        {
            double xsize = X.Count;
            double corr = 0;

            //multi thread
            List<dataSeqTest> X2 = new List<dataSeqTest>();
            multiThreading(X, X2);

            foreach (dataSeqTest x in X2)
            {
                //output tag results
                if (Global.swOutput != null)
                {
                    for (int i = 0; i < x._x.Count; i++)
                    {
                        Global.swOutput.Write(x._yOutput[i].ToString() + ",");
                    }
                    Global.swOutput.WriteLine();
                }

                List<int> goldTags = x._x.getTags();
                bool ck = true;
                for (int i = 0; i < x._x.Count; i++)
                {
                    if (goldTags[i] != x._yOutput[i])
                    {
                        ck = false;
                        break;
                    }
                }
                if (ck)
                    corr++;
            }
            double acc = corr / xsize * 100.0;
            Global.swLog.WriteLine("total-tag-strings={0}  correct-tag-strings={1}  string-accuracy={2}%", xsize, corr, acc);
            List<double> scoreList = new List<double>();
            scoreList.Add(acc);
            return scoreList;
        }

        //f-score
        public List<double> decode_fscore(dataSet X, model m, double iter)
        {
            //multi thread
            List<dataSeqTest> X2 = new List<dataSeqTest>();
            multiThreading(X, X2);

            List<string> goldTagList = new List<string>();
            List<string> resTagList = new List<string>();
           
            foreach (dataSeqTest x in X2)
            {
                string res = "";
                foreach (int im in x._yOutput)
                    res += im.ToString() + ",";
                resTagList.Add(res);

                //output tag results
                if (Global.swOutput != null)
                {
                    for (int i = 0; i < x._yOutput.Count; i++)
                    {
                        Global.swOutput.Write(x._yOutput[i] + ",");
                    }
                    Global.swOutput.WriteLine();
                }

                List<int> goldTags = x._x.getTags();
                string gold = "";
                foreach (int im in goldTags)
                    gold += im.ToString() + ",";
                goldTagList.Add(gold);
            }

            List<double> infoList = new List<double>();
            List<double> scoreList = fscore.getFscore(goldTagList, resTagList, infoList);
            Global.swLog.WriteLine("#gold-chunk={0}  #output-chunk={1}  #correct-output-chunk={2}  precision={3}  recall={4}  f-score={5}", infoList[0], infoList[1], infoList[2], scoreList[1].ToString("f2"), scoreList[2].ToString("f2"), scoreList[0].ToString("f2"));
            return scoreList;
        }

        public void multiThreading(dataSet X, List<dataSeqTest> X2)
        {
            //data for multi thread
            for (int i = 0; i < X.Count; i++)
                X2.Add(new dataSeqTest(X[i], new List<int>()));

            Global.threadXX = new List<List<dataSeqTest>>();
            for (int i = 0; i < Global.nThread; i++)
                Global.threadXX.Add(new List<dataSeqTest>());
            for (int i = 0; i < X2.Count; i++)
            {
                int idx = i % Global.nThread;
                Global.threadXX[idx].Add(X2[i]);
            }

            Stopwatch timer = new Stopwatch();
            timer.Start();

            //multi thread
            Task[] taskAry = new Task[Global.nThread];
            for (int i = 0; i < Global.nThread; i++)
            {
                taskAry[i] = new Task(taskRunner_test, i, TaskCreationOptions.PreferFairness);
                taskAry[i].Start();
            }

            Task.WaitAll(taskAry);

            timer.Stop();
            double time = timer.ElapsedMilliseconds / 1000.0;
            Global.swLog.WriteLine("**********test run time (sec): " + time.ToString());
        }

        //a thread
        public void taskRunner_test(object task)
        {
            int i = (int)task;

            for (int k = 0; k < Global.threadXX[i].Count; k++)
            {
                dataSeqTest x = Global.threadXX[i][k];
                //compute tags
                List<int> tags = new List<int>();
                double prob = _inf.decodeViterbi_fast(_model, x._x, tags);
                x._yOutput.Clear();
                foreach (int im in tags)
                    x._yOutput.Add(im);
            }
        }

        public model Model
        {
            get { return _model; }
        }

        public dataSet X
        {
            get { return _X; }
        }

        public inference Inf
        {
            get { return _inf; }
        }

        public gradient Grad
        {
            get { return _grad; }
        }

        public featureGenerator FGene
        {
            get { return _fGene; }
        }

        public optimizer Optim
        {
            get { return _optim; }
        }
    }
}