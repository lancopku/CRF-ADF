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
    class richEdge
    {
        public static double train()
        {
            //load data
            Console.WriteLine("\nreading training & test data...");
            Global.swLog.WriteLine("\nreading training & test data...");
            dataSet X, XX;
            if (Global.runMode.Contains("tune"))
            {
                dataSet origX = new dataSet(Global.fFeatureTrain, Global.fGoldTrain);
                X = new dataSet();
                XX = new dataSet();
                MainClass.dataSplit(origX, Global.tuneSplit, X, XX);
            }
            else
            {
                X = new dataSet(Global.fFeatureTrain, Global.fGoldTrain);
                XX = new dataSet(Global.fFeatureTest, Global.fGoldTest);
                MainClass.dataSizeScale(X);
            }
            Global.swLog.WriteLine("data sizes (train, test): {0} {1}", X.Count, XX.Count);

            double score = 0;
            foreach (double r in Global.regList)
            {
                Global.reg = r;
                Global.swLog.WriteLine("\nr: " + r.ToString());
                Console.WriteLine("\nr: " + r.ToString());
                if (Global.rawResWrite) Global.swResRaw.WriteLine("\n%r: " + r.ToString());
                toolboxRich tb = new toolboxRich(X);
                score = MainClass.basicTrain(XX, tb);
                resSummarize.write();
                //save model
                if (Global.save == 1)
                    tb.Model.save(Global.fModel);
            }
            return score;
        }

        public static double test()
        {
            dataSet X = new dataSet(Global.fFeatureTrain, Global.fGoldTrain);
            dataSet XX = new dataSet(Global.fFeatureTest, Global.fGoldTest);
            Global.swLog.WriteLine("data size (test): {0}", XX.Count);
            //load model for testing
            toolboxRich tb = new toolboxRich(X, false);

            List<double> scoreList = tb.test(XX, 0);

            double score = scoreList[0];
            Global.scoreListList.Add(scoreList);
            resSummarize.write();
            return score;
        }
    }

    class toolboxRich: toolbox
    {
        public toolboxRich(dataSet X, bool train = true)
        {
            if (train)//for training
            {
                _X = X;
                _fGene = new featureGeneRich(X);
                _model = new model(X, _fGene);
                _inf = new inferRich(this);
                _grad = new gradRich(this);
                initOptimizer();
            }
            else//for test
            {
                _X = X;
                _model = new model(Global.fModel);
                _fGene = new featureGeneRich(X);
                _inf = new inferRich(this);
                _grad = new gradRich(this);
            }
        }

    }

    class featureGeneRich: featureGenerator
    {
        protected int _nFeatureTemp_richEdge;
        protected int _backoff2;//end of non-rich edge feature

        public featureGeneRich()
        {
        }

        //for training & test
        public featureGeneRich(dataSet X)
        {
            _nFeatureTemp = X.NFeature;
            _nFeatureTemp_richEdge = (int)(X.NFeature * Global.edgeReduce);

            this._nTag = X.NTag;
            int nNodeFeature = _nFeatureTemp * _nTag;
            int nEdgeFeature1 = _nTag * _nTag;
            int nEdgeFeature2 = _nFeatureTemp_richEdge * _nTag * _nTag;

            _backoff1 = nNodeFeature;
            _backoff2 = nNodeFeature + nEdgeFeature1;
            _nCompleteFeature = nNodeFeature + nEdgeFeature1 + nEdgeFeature2;

            Global.swLog.WriteLine("feature templates & rich-edge feature templates0: {0}, {1}", _nFeatureTemp, _nFeatureTemp_richEdge);
            Global.swLog.WriteLine("nNodeFeature, nEdgeFeature1, nEdgeFeature2: {0}, {1}, {2}", nNodeFeature, nEdgeFeature1, nEdgeFeature2);
            Global.swLog.WriteLine("complete features: {0}", _nCompleteFeature);
            Global.swLog.WriteLine();
            Global.swLog.Flush();
        }

        override public int getEdgeFeatID(int id, int sPre, int s)
        {
            return _backoff2 + id * _nTag * _nTag + s * _nTag + sPre;
        }

        override public int getNRichFeatTemp()
        {
            return _nFeatureTemp_richEdge; 
        }
    }

    class inferRich: inference
    {
        public inferRich(toolbox tb)
            : base(tb)
        {
        }

        override public void getLogYY(model m, dataSeq x, int i, ref dMatrix YY, ref List<double> Y, bool takeExp, bool mask)
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
                    int f =_fGene.getNodeFeatID(ft.id,s);
                    Y[s] += w[f] * ft.val;
                }
            }
            if (i > 0)
            {
                //non-rich edge
                if (Global.useTraditionalEdge)
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

                //rich edge
                foreach (featureTemp im in fList)
                {
                    int id = im.id;
                    if (id < _fGene.getNRichFeatTemp())
                    {
                        for (int s = 0; s < nTag; s++)
                        {
                            for (int sPre = 0; sPre < nTag; sPre++)
                            {
                                int f = _fGene.getEdgeFeatID(id, sPre, s);
                                YY[sPre, s] += w[f] * im.val;
                            }
                        }
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

        //the scalar version
        override public void getLogYY(double scalar, model m, dataSeq x, int i, ref dMatrix YY, ref List<double> Y, bool takeExp, bool mask)
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
                    int f =_fGene.getNodeFeatID(ft.id,s);
                    Y[s] += w[f] * scalar * ft.val;
                }
            }
            if (i > 0)
            {
                //non-rich
                if (Global.useTraditionalEdge)
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

                //rich
                foreach (featureTemp im in fList)
                {
                    int id = im.id;
                    if (id < _fGene.getNRichFeatTemp())
                    {
                        for (int s = 0; s < nTag; s++)
                        {
                            for (int sPre = 0; sPre < nTag; sPre++)
                            {
                                int f = _fGene.getEdgeFeatID(id, sPre, s);
                                YY[sPre, s] += w[f] * scalar * im.val;
                            }
                        }
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
    }

    class gradRich: gradient
    {
        public gradRich(toolbox tb)
            : base(tb)
        {
        }

        override public double getGradCRF(List<double> gradList, model m, dataSeq x, baseHashSet<int> idSet)
        {
            if (idSet != null) idSet.Clear();
            int nTag = m.NTag;
            //compute beliefs
            belief bel = new belief(x.Count, nTag);
            belief belMasked = new belief(x.Count, nTag);
            //store the YY and Y
            List<dMatrix> YYlist = new List<dMatrix>(), maskYYlist = new List<dMatrix>();
            List<List<double>> Ylist = new List<List<double>>(), maskYlist = new List<List<double>>();
            _inf.getYYandY(m, x, YYlist, Ylist, maskYYlist, maskYlist);
            _inf.getBeliefs(bel, m, x, YYlist, Ylist);
            _inf.getBeliefs(belMasked, m, x, maskYYlist, maskYlist);
            double ZGold = belMasked.Z;
            double Z = bel.Z;

            List<featureTemp> fList;
            //Loop over nodes to compute features and update the gradient
            for (int i = 0; i < x.Count; i++)
            {
                fList = _fGene.getFeatureTemp(x, i);
                foreach(featureTemp im in fList)
                {
                    for (int s = 0; s < nTag; s++)
                    {
                        int f = _fGene.getNodeFeatID(im.id, s);
                        if (idSet != null) idSet.Add(f);
                        
                        gradList[f] += bel.belState[i][s] * im.val;
                        gradList[f] -= belMasked.belState[i][s] * im.val;
                    }
                }
            }

            //Loop over edges to compute features and update the gradient
            for (int i = 1; i < x.Count; i++) 
            {
                //non-rich
                if (Global.useTraditionalEdge)
                {
                    for (int s = 0; s < nTag; s++)
                    {
                        for (int sPre = 0; sPre < nTag; sPre++)
                        {
                            int f = _fGene.getEdgeFeatID(sPre, s);
                            if (idSet != null) idSet.Add(f);

                            gradList[f] += bel.belEdge[i][sPre, s];
                            gradList[f] -= belMasked.belEdge[i][sPre, s];
                        }
                    }
                }

                //rich
                fList = _fGene.getFeatureTemp(x, i);
                foreach (featureTemp im in fList)
                {
                    int id = im.id;
                    if (id < _fGene.getNRichFeatTemp())
                    {
                        for (int s = 0; s < nTag; s++)
                        {
                            for (int sPre = 0; sPre < nTag; sPre++)
                            {
                                int f = _fGene.getEdgeFeatID(id, sPre, s);
                                if (idSet != null) idSet.Add(f);

                                gradList[f] += bel.belEdge[i][sPre, s] * im.val;
                                gradList[f] -= belMasked.belEdge[i][sPre, s] * im.val;
                            }
                        }
                    }
                }
            }
            return Z - ZGold;//-log{P(y*|x,w)}
        }

        //this is the scalar version
        override public double getGradCRF(List<double> gradList, double scalar, model m, dataSeq x, baseHashSet<int> idSet)
        {
            idSet.Clear();
            int nTag = m.NTag;
            //compute beliefs
            belief bel = new belief(x.Count, nTag);
            belief belMasked = new belief(x.Count, nTag);
            _inf.getBeliefs(bel,m, x, scalar, false);
            _inf.getBeliefs(belMasked,m, x, scalar, true);
            double ZGold = belMasked.Z;
            double Z = bel.Z;

            List<featureTemp> fList;
            //Loop over nodes to compute features and update the gradient
            for (int i = 0; i < x.Count; i++)
            {
                fList = _fGene.getFeatureTemp(x, i);
                foreach(featureTemp im in fList)
                {
                    for (int s = 0; s < nTag; s++)
                    {
                        int f = _fGene.getNodeFeatID(im.id, s);
                        idSet.Add(f);
                        
                        gradList[f] += bel.belState[i][s] * im.val;
                        gradList[f] -= belMasked.belState[i][s] * im.val;
                    }
                }
            }

            //Loop over edges to compute features and update the gradient
            for (int i = 1; i < x.Count; i++) 
            {
                //non-rich
                if (Global.useTraditionalEdge)
                {
                    for (int s = 0; s < nTag; s++)
                    {
                        for (int sPre = 0; sPre < nTag; sPre++)
                        {
                            int f = _fGene.getEdgeFeatID(sPre, s);
                            idSet.Add(f);

                            gradList[f] += bel.belEdge[i][sPre, s];
                            gradList[f] -= belMasked.belEdge[i][sPre, s];
                        }
                    }
                }

                //rich
                fList = _fGene.getFeatureTemp(x, i);
                foreach (featureTemp im in fList)
                {
                    int id = im.id;
                    if (id < _fGene.getNRichFeatTemp())
                    {
                        for (int s = 0; s < nTag; s++)
                        {
                            for (int sPre = 0; sPre < nTag; sPre++)
                            {
                                int f = _fGene.getEdgeFeatID(id, sPre, s);
                                idSet.Add(f);

                                gradList[f] += bel.belEdge[i][sPre, s] * im.val;
                                gradList[f] -= belMasked.belEdge[i][sPre, s] * im.val;
                            }
                        }
                    }
                }
            }
            return Z - ZGold;//-log{P(y*|x,w)}
        }
    }
}