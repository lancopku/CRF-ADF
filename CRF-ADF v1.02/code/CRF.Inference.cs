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

namespace Program
{
    class belief
    {
        public List<List<double>> belState;
        public List<dMatrix> belEdge;
        public double Z;

        public belief(int nNodes, int nStates)
        {
            List<double>[] dAry = new List<double>[nNodes];
            belState = new List<List<double>>(dAry);
            for (int i = 0; i < nNodes; i++)
            {
                double[] dAry2 = new double[nStates];
                belState[i] = new List<double>(dAry2);
            }

            dMatrix[] dAry3 = new dMatrix[nNodes];
            belEdge = new List<dMatrix>(dAry3);
            for (int i = 1; i < nNodes; i++)
                belEdge[i] = new dMatrix(nStates, nStates);

            belEdge[0] = null;
            Z = 0;
        }
    }

    class inference
    {
        protected optimizer _optim;
        protected featureGenerator _fGene;
        protected gradient _grad;

        public inference(toolbox tb)
        {
            _optim = tb.Optim;
            _fGene = tb.FGene;
            _grad = tb.Grad;
        }

        public void getYYandY(model m, dataSeq x, List<dMatrix> YYlist, List<List<double>> Ylist, List<dMatrix> maskYYlist, List<List<double>> maskYlist)
        {
            int nNodes = x.Count;
            int nTag = m.NTag;
            double[] dAry = new double[nTag];
            bool mask = false;

            try
            {
                //Global.rwlock.AcquireReaderLock(Global.readWaitTime);

                for (int i = 0; i < nNodes; i++)
                {
                    dMatrix YYi = new dMatrix(nTag, nTag);
                    List<double> Yi = new List<double>(dAry);
                    //compute the Mi matrix
                    getLogYY(m, x, i, ref YYi, ref Yi, false, mask);
                    YYlist.Add(YYi);
                    Ylist.Add(Yi);

                    maskYYlist.Add(new dMatrix(YYi));
                    maskYlist.Add(new List<double>(Yi));
                }

                //Global.rwlock.ReleaseReaderLock();
            }
            catch (ApplicationException)
            {
                Console.WriteLine("read out time!");
            }

            //get the masked YY and Y
            double maskValue = double.MinValue;
            for (int i = 0; i < nNodes; i++)
            {
                List<double> Y = maskYlist[i];
                List<int> tagList = x.getTags();
                for (int s = 0; s < Y.Count; s++)
                {
                    if (tagList[i] != s)
                        Y[s] = maskValue;
                }
            }
        }

        //get beliefs (mariginal probabilities)
        public void getBeliefs(belief bel, model m, dataSeq x, List<dMatrix> YYlist, List<List<double>> Ylist)
        {
            int nNodes = x.Count;
            int nTag = m.NTag;

            //dMatrix YY = new dMatrix(nTag, nTag);
            double[] dAry = new double[nTag];
            //List<double> Y = new List<double>(dAry);
            List<double> alpha_Y = new List<double>(dAry);
            List<double> newAlpha_Y = new List<double>(dAry);//marginal probability from left to current node (including values of the current node)
            List<double> tmp_Y = new List<double>(dAry);

            //compute beta values in a backward scan
            for (int i = nNodes - 1; i > 0; i--)
            {
                dMatrix YY = YYlist[i];
                List<double> Y = Ylist[i];
                //compute the Mi matrix
                //getLogYY(m, x, i, ref YY, ref Y, false, mask);
                listTool.listSet(ref tmp_Y, bel.belState[i]);//this is meaningful from the 2nd round
                listTool.listAdd(ref tmp_Y, Y);
                logMultiply(YY, tmp_Y, bel.belState[i - 1]);
            }
            //compute alpha values
            for (int i = 0; i < nNodes; i++)
            {
                dMatrix YY = null;
                if (i > 0)
                    YY = new dMatrix(YYlist[i]);//should use the copy to avoid change
                List<double> Y = Ylist[i];
                //compute the Mi matrix
                //getLogYY(m, x, i, ref YY, ref Y, false, mask);
                if (i > 0)
                {
                    listTool.listSet(ref tmp_Y, alpha_Y);//this is meaningful from the 2nd round
                    YY.transpose();
                    logMultiply(YY, tmp_Y, newAlpha_Y);
                    listTool.listAdd(ref newAlpha_Y, Y);
                }
                else
                {
                    listTool.listSet(ref newAlpha_Y, Y);
                }
                //setting marginal probability on edges
                if (i > 0)
                {
                    //beta + Y
                    listTool.listSet(ref tmp_Y, Y);
                    listTool.listAdd(ref tmp_Y, bel.belState[i]);
                    //YY
                    YY.transpose();
                    bel.belEdge[i].set(YY);
                    //belief = alpha + YY + beta + Y
                    for (int yPre = 0; yPre < nTag; yPre++)
                        for (int y = 0; y < nTag; y++)
                            bel.belEdge[i][yPre, y] += tmp_Y[y] + alpha_Y[yPre];
                }
                //setting marginal probability on nodes
                List<double> tmp = bel.belState[i];//beta
                listTool.listAdd(ref tmp, newAlpha_Y);//belief = alpha + beta
                listTool.listSet(ref alpha_Y, newAlpha_Y);
            }
            double Z = logSum(alpha_Y);
            for (int i = 0; i < nNodes; i++)
            {
                List<double> tmp = bel.belState[i];
                listTool.listAdd(ref tmp, -Z);
                listTool.listExp(ref tmp);
            }
            for (int i = 1; i < nNodes; i++)
            {
                bel.belEdge[i].add(-Z);
                bel.belEdge[i].eltExp();
            }
            bel.Z = Z;//the overall potential function value
        }

        //the scalar version
        public void getBeliefs(belief bel, model m, dataSeq x, double scalar, bool mask)
        {
            int nNodes = x.Count;
            int nTag = m.NTag;

            dMatrix YY = new dMatrix(nTag, nTag);
            double[] dAry = new double[nTag];
            List<double> Y = new List<double>(dAry);
            List<double> alpha_Y = new List<double>(dAry);
            List<double> newAlpha_Y = new List<double>(dAry);
            List<double> tmp_Y = new List<double>(dAry);
            //compute beta values in a backward scan
            for (int i = nNodes - 1; i > 0; i--)
            {
                getLogYY(scalar, m,x, i, ref YY, ref Y, false, mask);
                listTool.listSet(ref tmp_Y, bel.belState[i]);
                listTool.listAdd(ref tmp_Y, Y);
                logMultiply(YY, tmp_Y, bel.belState[i - 1]);
            }
            //compute alpha values
            for (int i = 0; i < nNodes; i++)
            {
                getLogYY(scalar, m,x, i, ref YY, ref Y, false, mask);
                if (i > 0)
                {
                    listTool.listSet(ref tmp_Y, alpha_Y);
                    YY.transpose();
                    logMultiply(YY, tmp_Y, newAlpha_Y);
                    listTool.listAdd(ref newAlpha_Y, Y);
                }
                else
                {
                    listTool.listSet(ref newAlpha_Y, Y);
                }
                if (i > 0)
                {
                    listTool.listSet(ref tmp_Y, Y);
                    listTool.listAdd(ref tmp_Y, bel.belState[i]);
                    YY.transpose();
                    bel.belEdge[i].set(YY);
                    for (int yPre = 0; yPre < nTag; yPre++)
                        for (int y = 0; y < nTag; y++)
                            bel.belEdge[i][yPre, y] += tmp_Y[y] + alpha_Y[yPre];
                }
                List<double> tmp = bel.belState[i];
                listTool.listAdd(ref tmp, newAlpha_Y);
                listTool.listSet(ref alpha_Y, newAlpha_Y);
            }
            double Z = logSum(alpha_Y);
            for (int i = 0; i < nNodes; i++)
            {
                List<double> tmp = bel.belState[i];
                listTool.listAdd(ref tmp, -Z);
                listTool.listExp(ref tmp);
            }
            for (int i = 1; i < nNodes; i++)
            {
                bel.belEdge[i].add(-Z);
                bel.belEdge[i].eltExp();
            }
            bel.Z = Z;
        }

        virtual public void getLogYY(model m, dataSeq x, int i, ref dMatrix YY, ref List<double> Y, bool takeExp, bool mask)
        {
            YY.set(0);
            listTool.listSet(ref Y, 0);

            float[] w = m.W;
            List<featureTemp> fList = _fGene.getFeatureTemp(x, i);
            int nTag = m.NTag;
            foreach(featureTemp ft in fList)
            {
                for (int s = 0; s < nTag; s++)
                {
                    int f = _fGene.getNodeFeatID(ft.id, s);
                    Y[s] += w[f] * ft.val;
                }
            }
            if (i > 0)
            {
                for (int s = 0; s < nTag; s++)
                {
                    for (int sPre = 0; sPre < nTag; sPre++)
                    {
                        int f = _fGene.getEdgeFeatID(sPre, s);
                        YY[sPre, s] += w[f];
                    }
                }
            }
            double maskValue = double.MinValue;
            if (takeExp)
            {
                listTool.listExp(ref Y);
                YY.eltExp();
                maskValue = 0;
            }
            if (mask)
            {
                List<int> tagList = x.getTags();
                for (int s = 0; s < Y.Count; s++)
                {
                    if (tagList[i]!=s)
                        Y[s] = maskValue;
                }
            }
        }

        //the scalar version
        virtual public void getLogYY(double scalar, model m, dataSeq x, int i, ref dMatrix YY, ref List<double> Y, bool takeExp, bool mask)
        {
            YY.set(0);
            listTool.listSet(ref Y, 0);

            float[] w = m.W;
            List<featureTemp> fList = _fGene.getFeatureTemp(x, i);
            int nTag = m.NTag;
            foreach(featureTemp ft in fList)
            {
                for (int s = 0; s < nTag; s++)
                {
                    int f = _fGene.getNodeFeatID(ft.id, s);
                    Y[s] += w[f] * scalar * ft.val;
                }
            }
            if (i > 0)
            {
                for (int s = 0; s < nTag; s++)
                {
                    for (int sPre = 0; sPre < nTag; sPre++)
                    {
                        int f = _fGene.getEdgeFeatID(sPre, s);
                        YY[sPre, s] += w[f] * scalar;
                    }
                }
            }
            double maskValue = double.MinValue;
            if (takeExp)
            {
                listTool.listExp(ref Y);
                YY.eltExp();
                maskValue = 0;
            }
            if (mask)
            {
                List<int> tagList = x.getTags();
                for (int s = 0; s < Y.Count; s++)
                {
                    if (tagList[i] != s)
                        Y[s] = maskValue;
                }
            }
        }

        public double decodeViterbi_fast(model m, dataSeq x, List<int> tags)
        {
            tags.Clear();
            int nNode = x.Count;
            int nTag = m.NTag;
            dMatrix YY = new dMatrix(nTag, nTag);
            double[] dAry = new double[nTag];
            List<double> Y = new List<double>(dAry);
            Viterbi viter = new Viterbi(nNode, nTag);

            for (int i = 0; i < nNode; i++)
            {
                getLogYY(m, x, i, ref YY, ref Y, false, false);
                viter.setScores(i, Y, YY);
            }

            double numer = viter.runViterbi(ref tags, false);
            return numer;
        }

        //compute the multiply-results of a matrix A and a vertical list B based on log-sum: A*B
        public void logMultiply(dMatrix A, List<double> B, List<double> AB)
        {
            List<double>[] toSumLists = new List<double>[A.R];
            for (int i = 0; i < toSumLists.Length; i++)
                toSumLists[i] = new List<double>();

            for (int r = 0; r < A.R; r++)
                for (int c = 0; c < A.C; c++)
                    toSumLists[r].Add(A[r, c] + B[c]);

            for (int r = 0; r < A.R; r++)
            {
                AB[r] = logSum(toSumLists[r]);
            }
        }

        //summing based on log-level, it need a trick of e^x+e^y=e^[x+log(1+e^(y-x)]
        public static double logSum(List<double> a)
        {
            double m1;
            double m2;
            double sum = a[0];
            for (int i = 1; i < a.Count; i++)
            {
                if (sum >= a[i])
                {
                    m1 = sum;
                    m2 = a[i];
                }
                else
                {
                    m1 = a[i];
                    m2 = sum;
                }
                sum = m1 + Math.Log(1 + Math.Exp(m2 - m1));
            }
            return sum;
        }
    }
}