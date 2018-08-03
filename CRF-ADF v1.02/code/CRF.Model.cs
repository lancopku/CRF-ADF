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
    class model
    {
        int _nTag;
        float[] _w;

        public model(string file)
        {
            if(File.Exists(file))
                load(file);
        }

        public model(dataSet X, featureGenerator fGen)
        {
            _nTag = X.NTag;
            //default value is 0
            if (Global.random == 0)
            {
                _w = new float[fGen.NCompleteFeature];
            }
            else if (Global.random == 1)
            {
                List<float> randList = randomDoubleTool.getRandomList_float(fGen.NCompleteFeature);
                _w = randList.ToArray();
            }
            else throw new Exception("error");
        }

        public model(model m, bool wCopy)
        {
            _nTag = m.NTag;
            _w = new float[m.W.Length];
            if (wCopy)
            {
                m.W.CopyTo(_w, 0);
            }
        }

        public void load(string file)
        {
            StreamReader sr = new StreamReader(file);
            string txt = sr.ReadToEnd();
            txt = txt.Replace("\r", "");
            string[] ary = txt.Split(Global.lineEndAry, StringSplitOptions.RemoveEmptyEntries);
            _nTag = int.Parse(ary[0]);
            int wsize = int.Parse(ary[1]);
            _w = new float[wsize];
            for (int i = 2; i < ary.Length; i++)
            {
                _w[i - 2] = float.Parse(ary[i]);
            }

            sr.Close();
        }

        public void save(string file)
        {
            StreamWriter sw = new StreamWriter(file);
            sw.WriteLine(_nTag);
            sw.WriteLine(_w.Length);
            foreach (float im in _w)
            {
                sw.WriteLine(im.ToString("f4"));
            }
            sw.Close();
        }

        public float[] W
        {
            get { return _w; }
            set 
            {
                if (_w == null)
                {
                    float[] ary = new float[value.Length];
                }
                value.CopyTo(_w, 0);
            }                 
        }

        public int NTag
        {
            get { return _nTag; }
            set
            {
                _nTag = value;
            }
        }

    }
}
