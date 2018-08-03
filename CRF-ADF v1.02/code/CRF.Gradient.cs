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
    class gradient
    {
        protected optimizer _optim;
        protected inference _inf;
        protected featureGenerator _fGene;

        public gradient(toolbox tb)
        {
            _optim = tb.Optim;
            _inf = tb.Inf;
            _fGene = tb.FGene;
        }

        virtual public double getGradCRF(List<double> vecGrad, model m, dataSeq x, baseHashSet<int> idSet)
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
                        int f = _fGene.getNodeFeatID(im.id,s);
                        if (idSet != null) idSet.Add(f);
                        
                        vecGrad[f] += bel.belState[i][s] * im.val;
                        vecGrad[f] -= belMasked.belState[i][s] * im.val;
                    }
                }
            }

            //Loop over edges to compute features and update the gradient
            for (int i = 1; i < x.Count; i++) 
            {
                for (int s = 0; s < nTag; s++)
                {
                    for (int sPre = 0; sPre < nTag; sPre++)
                    {
                        int f = _fGene.getEdgeFeatID(sPre, s);
                        if (idSet != null) idSet.Add(f);
                        
                        vecGrad[f] += bel.belEdge[i][sPre, s];
                        vecGrad[f] -= belMasked.belEdge[i][sPre, s];
                    }
                }
            }
            return Z - ZGold;//-log{P(y*|x,w)}
        }

        //the scalar version
        virtual public double getGradCRF(List<double> vecGrad, double scalar, model m, dataSeq x, baseHashSet<int> idSet)
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
                        int f =_fGene.getNodeFeatID(im.id,s);
                        idSet.Add(f);
                        
                        vecGrad[f] += bel.belState[i][s] * im.val;
                        vecGrad[f] -= belMasked.belState[i][s] * im.val;
                    }
                }
            }

            //Loop over edges to compute features and update the gradient
            for (int i = 1; i < x.Count; i++) 
            {
                for (int s = 0; s < nTag; s++)
                {
                    for (int sPre = 0; sPre < nTag; sPre++)
                    {
                        int f = _fGene.getEdgeFeatID(sPre, s);
                        idSet.Add(f);
                        
                        vecGrad[f] += bel.belEdge[i][sPre, s];
                        vecGrad[f] -= belMasked.belEdge[i][sPre, s];
                    }
                }
            }
            return Z - ZGold;//-log{P(y*|x,w)}
        }

        public double getGrad_SGD(List<double> g, model m, dataSeq x, baseHashSet<int> idset)
        {
            if (idset != null) 
                idset.Clear();

            if (x == null)
                return 0;

            return getGradCRF(g, m, x, idset);
        }

        //the scalar version
        public double getGrad_SGD(List<double> g, double scalar, model m, dataSeq x, baseHashSet<int> idset)
        {
            return getGradCRF(g, scalar, m, x, idset);
        }

        //the mini-batch version
        public double getGrad_SGD_miniBatch(List<double> g, model m, List<dataSeq> X, baseHashSet<int> idset)
        {
            if (idset != null) idset.Clear();
            double error = 0;
            foreach (dataSeq x in X)
            {
                baseHashSet<int> idset2 = new baseHashSet<int>();
                
                error += getGradCRF(g, m, x, idset2);

                if (idset != null)
                {
                    foreach (int i in idset2)
                        idset.Add(i);
                }
            }
            return error;
        }

        //compute grad of: sum{-log{P(y*|x,w)}} + R(w)
        public double getGrad_BFGS(List<double> g, model m, dataSet X)
        {
            double error = 0;
            int nFeature = _fGene.NCompleteFeature;

            foreach (dataSeq x in X)
            {
                double err = 0;
                err = getGradCRF(g, m, x, null);
                error += err;
            }

            if (Global.reg != 0.0)
            {
                for (int f = 0; f < nFeature; f++)
                {
                    g[f] += m.W[f] / (Global.reg * Global.reg);
                }
            }
            if (Global.reg != 0.0)
            {
                float[] tmpWeights = m.W;
                double sum = arrayTool.squareSum(tmpWeights);
                error += sum / (2.0 * Global.reg * Global.reg);
            }
            return error;
        }

    }
}