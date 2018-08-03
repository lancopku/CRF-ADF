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
    class optimStochastic : optimizer
    {
        public optimStochastic(toolbox tb)
        {
            _model = tb.Model;
            _X = tb.X;
            _inf = tb.Inf;
            _fGene = tb.FGene;
            _grad = tb.Grad;
            //init
            int fsize = _model.W.Length;
            Global.decayList = new List<double>(new double[fsize]);
            listTool.listSet(ref Global.decayList, Global.rate0);
        }

        override public double optimize()
        {
            double error = 0;
            if (Global.modelOptimizer.EndsWith("adf"))
                error = adf();
            else if (Global.modelOptimizer.EndsWith("sgder"))
                error = sgd_exactReg();
            else    
                error = sgd_lazyReg();

            Global.swLog.Flush();
            return error;
        }

        //ADF training
        public double adf()
        {
            float[] w = _model.W;
            int fsize = w.Length;
            int xsize = _X.Count;
            List<double> grad = new List<double>(new double[fsize]);
            double error = 0;
            List<int> featureCountList = new List<int>(new int[fsize]);
            List<int> ri = randomTool<int>.getShuffledIndexList(xsize);//random shuffle of training samples
            Global.interval = xsize / Global.nUpdate;
            int nSample = 0;//#sample in an update interval

            for (int t = 0; t < xsize; t += Global.miniBatch)
            {
                List<dataSeq> XX = new List<dataSeq>();
                bool end = false;
                for (int k = t; k < t + Global.miniBatch; k++)
                {
                    int i = ri[k];
                    dataSeq x = _X[i];
                    XX.Add(x);
                    if (k == xsize - 1)
                    {
                        end = true;
                        break;
                    }
                }
                int mbSize = XX.Count;
                nSample += mbSize;
                baseHashSet<int> fSet = new baseHashSet<int>();
                double err = _grad.getGrad_SGD_miniBatch(grad, _model, XX, fSet);
                error += err;

                foreach (int i in fSet)
                    featureCountList[i]++;

                bool check = false;
                for (int k = t; k < t + Global.miniBatch; k++)
                {
                    if (t != 0 && k % Global.interval == 0)
                        check = true;
                }
                //update decay rates
                if (check || end)
                {
                    for (int i = 0; i < fsize; i++)
                    {
                        int v = featureCountList[i];
                        double u = (double)v / (double)nSample;
                        double eta = Global.upper - (Global.upper - Global.lower) * u;
                        Global.decayList[i] *= eta;
                    }
                    //reset
                    for (int i = 0; i < featureCountList.Count; i++)
                        featureCountList[i] = 0;
                }
                //update weights
                foreach (int i in fSet)
                {
                    w[i] -= (float)(Global.decayList[i] * grad[i]);
                    //reset
                    grad[i] = 0;
                }
                //reg
                if (check || end)
                {
                    if (Global.reg != 0)
                    {
                        for (int i = 0; i < fsize; i++)
                        {
                            double grad_i = w[i] / (Global.reg * Global.reg) * ((double)nSample / (double)xsize);
                            w[i] -= (float)(Global.decayList[i] * grad_i);
                        }
                    }
                    //reset
                    nSample = 0;
                }
                Global.countWithIter += mbSize;
            }

            if (Global.reg != 0)
            {
                double sum = arrayTool.squareSum(w);
                error += sum / (2.0 * Global.reg * Global.reg);
            }

            Global.diff = convergeTest(error);
            return error;
        }

        //SGD with lazy reg
        public double sgd_lazyReg()
        {
            float[] w = _model.W;
            int fsize = w.Length;
            int xsize = _X.Count;
            double[] ary = new double[fsize];
            List<double> grad = new List<double>(ary);

            List<int> ri = randomTool<int>.getShuffledIndexList(xsize);
            double error = 0;
            double r_k = 0;

            for (int t = 0; t < xsize; t += Global.miniBatch)
            {
                List<dataSeq> XX = new List<dataSeq>();
                for (int k = t; k < t + Global.miniBatch; k++)
                {
                    int i = ri[k];
                    dataSeq x = _X[i];
                    XX.Add(x);
                    if (k == xsize - 1)
                        break;
                }
                int mbSize = XX.Count;
                baseHashSet<int> fset = new baseHashSet<int>();
                double err = _grad.getGrad_SGD_miniBatch(grad, _model, XX, fset);
                error += err;

                //decaying rate: r_k = r_0 * beta^(k/N), with 0<r_0<=1, 0<beta<1 
                r_k = Global.rate0 * Math.Pow(Global.decayFactor, (double)Global.countWithIter / (double)xsize);

                if (Global.countWithIter % (xsize / 4) == 0)
                    Global.swLog.WriteLine("iter{0}    decay_rate={1}", Global.glbIter, r_k.ToString("e2"));

                foreach (int i in fset)
                {
                    //because dgrad[i] is the grad of -log(obj), minus the gradient to find the minumum point
                    w[i] -= (float)(r_k * grad[i]);
                    //reset
                    grad[i] = 0;
                }
                Global.countWithIter += mbSize;
            }

            if (Global.reg != 0)
            {
                for (int i = 0; i < fsize; i++)
                {
                    double grad_i = w[i] / (Global.reg * Global.reg);
                    w[i] -= (float)(r_k * grad_i);
                }

                double sum = arrayTool.squareSum(w);
                error += sum / (2.0 * Global.reg * Global.reg);
            }

            Global.diff = convergeTest(error);
            return error;
        }

        public double sgd_exactReg()
        {
            double scalar = 1, scalarOld = 1;
            float[] w = _model.W;
            int fsize = w.Length;
            int xsize = _X.Count;
            double newReg = Global.reg * Math.Sqrt(xsize);
            double oldReg = Global.reg;
            Global.reg = newReg;

            double[] tmpAry = new double[fsize];
            List<double> grad = new List<double>(tmpAry);

            List<int> ri = randomTool<int>.getShuffledIndexList(xsize);
            double error = 0;
            double r_k = 0;

            for (int t = 0; t < xsize; t++)
            {
                int ii = ri[t];
                dataSeq x = _X[ii];
                baseHashSet<int> fset = new baseHashSet<int>();
                double err = _grad.getGrad_SGD(grad, scalar, _model, x, fset);
                error += err;
                //decaying rate: r_k = r_0 * beta^(k/N), with 0<r_0<=1, 0<beta<1 
                r_k = Global.rate0 * Math.Pow(Global.decayFactor, (double)Global.countWithIter / (double)xsize);
                if (Global.countWithIter % (xsize / 4) == 0)
                    Global.swLog.WriteLine("iter{0}    decay_rate={1}", Global.glbIter, r_k.ToString("e2"));

                //reg
                if (t % Global.scalarResetStep == 0)
                {
                    //reset
                    for (int i = 0; i < fsize; i++)
                        w[i] *= (float)scalar;
                    scalar = scalarOld = 1;
                }
                else
                {
                    scalarOld = scalar;
                    scalar *= 1 - r_k / (Global.reg * Global.reg);
                }

                foreach (int i in fset)
                {
                    double realWeight = w[i] * scalarOld;
                    double grad_i = grad[i] + realWeight / (Global.reg * Global.reg);
                    realWeight = realWeight - r_k * grad_i;
                    w[i] = (float)(realWeight / scalar);

                    //reset
                    grad[i] = 0;
                }
                Global.countWithIter++;
            }

            //recover the real weights
            for (int i = 0; i < fsize; i++)
            {
                w[i] *= (float)scalar;
            }

            if (Global.reg != 0.0)
            {
                double sum = arrayTool.squareSum(w);
                error += sum / (2.0 * Global.reg * Global.reg);
            }

            Global.diff = convergeTest(error);
            Global.reg = oldReg;
            return error;
        }

    }
}