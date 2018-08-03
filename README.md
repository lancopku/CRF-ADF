# CRF-ADF Sequential Tagging Toolkit v1.0
This is a general purpose software for sequential tagging (or called sequential labelling, linear-chain structured classification). The CRF (Conditional Random Fields) model is described in (Lafferty et al., 2001) and the ADF (Adaptive stochastic gradient Decent based on Feature-frequency information) fast training algorithm is described in (Sun et al., ACL 2012). [[Tutorial]](CA.tu.pdf)

Main features:

 - Developed with C#
 - High accuracy (72.3% on Bio-Entity Recognition Task at BioNLP/NLPBA 2004, and 97.5% on Chinese Word Segmentation MSR Task)
 - Fast training (faster convergence rate than traditional batch/online training methods, including LBFGS & SGD)
 - General purpose (it is task-independent & trainable using your own tagged corpus)
 - Support rich edge features (Sun et al., ACL 2012)
 - Support various training methods, including ADF training, SGD training, & Limited-memory BFGS training
 - Support automatic n-fold cross-validation for tuning hyper-parameters
 - Support various evaluation metrics, including token-accuracy, string-accuracy, & F-score
