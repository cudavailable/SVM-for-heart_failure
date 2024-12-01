## SVM-for-heart_failure
任务：在UCI提供的心脏衰竭临床数据集上训练支持向量机模型，帮助根据患者特征进行生存情况预测

## 模型配置
- kernel_type : linear
- C : 3.0
- epsilon : 1e-6

## 10折交叉验证评估结果
- Accuracy: 0.5552
- Precision: 0.5755
- Recall: 0.8333
- F1-Score: 0.6809
- 具体训练情况和评估结果可参考log文件夹中的日志文件

## 使用说明
1. 克隆本项目代码到本地；
2. 先运行main函数中的valid函数，且不运行train函数，通过10折交叉验证确定较好的模型超参数（主要为前述模型配置中所提到的超参）；
3. 注释valid函数，运行train函数在全数据集上进行训练（已有保存模型）；  
4. 可根据已保存模型进行预测。这部分只写了加载模型函数，预测函数可自行构建。注意预测的输入数据需要和原始数据集格式一致。  

## 参考文献
1. Heart Failure Clinical Records [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5Z89R.  
2. https://github.com/LasseRegin/SVM-w-SMO  
3. 数据来源(http://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)  
