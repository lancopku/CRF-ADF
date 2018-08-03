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
    //feature template
    struct featureTemp
    {
        public readonly int id;//feature id
        public readonly double val;//feature value

        public featureTemp(int a, double b)
        {
            id = a;
            val = b;
        }
    }

    class featureGenerator
    {
        protected int _nFeatureTemp;
        protected int _nCompleteFeature;
        protected int _backoff1;//end of node feature
        protected int _nTag;

        public featureGenerator()
        {
        }

        //for train & test
        public featureGenerator(dataSet X)
        {
            _nFeatureTemp = X.NFeature;
            _nTag = X.NTag;
            Global.swLog.WriteLine("feature templates: {0}", _nFeatureTemp);

            int nNodeFeature = _nFeatureTemp * _nTag;
            int nEdgeFeature = _nTag * _nTag;
            _backoff1 = nNodeFeature;
            _nCompleteFeature = nNodeFeature + nEdgeFeature;
            Global.swLog.WriteLine("complete features: {0}", _nCompleteFeature);
        }

        public List<featureTemp> getFeatureTemp(dataSeq x, int node)
        {
            return x.getFeatureTemp(node);
        }

        public int getNodeFeatID(int id, int s)
        {
            return id * _nTag + s;
        }

        virtual public int getEdgeFeatID(int sPre, int s)
        {
            return _backoff1 + s * _nTag + sPre;
        }

        virtual public int getEdgeFeatID(int id, int sPre, int s)
        {
            throw new Exception("error");
        }

        virtual public void getFeatures(dataSeq x, int node, ref List<List<int>> nodeFeature, ref int[,] edgeFeature)
        {
            throw new Exception("error");
        }

        virtual public int getNRichFeatTemp()
        {
            throw new Exception("error");
        }

        public int Backoff1 { get { return _backoff1; } }

        public int NCompleteFeature { get { return _nCompleteFeature; } }
    }
}