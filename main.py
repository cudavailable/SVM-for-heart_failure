import os
import argparse
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from data import getData
from SVM import SVM
from logger import Logger

def save_mode(model_path, model):
    joblib.dump(model, model_path)

def load_model(model_path):
    return joblib.load(model_path)

def train(args):
    # 数据准备
    X, Y = getData(args.data_path)

    # 模型初始化
    model = SVM(kernel_type=args.kernel_type, C=args.C, epsilon=args.epsilon)

    # 保存日志
    if (args.log_dir is not None and not os.path.exists(args.log_dir)):
        os.mkdir(args.log_dir)
    train_logger = Logger(os.path.join(args.log_dir, args.train_log_path))

    train_logger.write(f"Train info\n-----------------\n")
    train_logger.write(f"max_iter: {model.max_iter}\n")
    train_logger.write(f"kernel_type: {model.kernel_type}\n")
    train_logger.write(f"C: {model.C}\n")
    train_logger.write(f"epsilon: {model.epsilon}\n\n")

    # 训练模型
    support_vectors, iter = model.fit(X, Y)
    sv_count = support_vectors.shape[0]

    # 保存模型
    save_mode(args.model_path, model)

    train_logger.write(f"Training completed.\n\n")

    train_logger.write(f"Support vector count: {sv_count}\n")
    train_logger.write(f"bias: {model.b:.3f}\n")
    train_logger.write(f"w: {model.w}\n")
    train_logger.write(f"Converged after {iter} iterations\n")
    train_logger.write(f"SVM model saved to {args.model_path}\n")

def valid(args):
    # 数据准备
    X, y = getData(args.data_path)

    # 保存日志
    if (args.log_dir is not None and not os.path.exists(args.log_dir)):
        os.mkdir(args.log_dir)
    valid_logger = Logger(os.path.join(args.log_dir, args.valid_log_path))
    valid_logger.write(f"Valid info\n-----------------\n")
    valid_logger.write(f"kernel_type: {args.kernel_type}\n")
    valid_logger.write(f"C: {args.C}\n")
    valid_logger.write(f"epsilon: {args.epsilon}\n\n")

    # 存储所有折的预测结果
    y_true = []
    y_pred = []

    # 10折交叉验证
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 初始化SVM模型
        svm = SVM(kernel_type=args.kernel_type, C=args.C, epsilon=args.epsilon)

        # 训练SVM模型
        svm.fit(X_train, y_train)

        # 在测试集上进行预测
        y_test_pred = svm.predict(X_test)

        # 存储实际标签和预测标签
        y_true.extend(y_test)
        y_pred.extend(y_test_pred)

    # 计算评估指标
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')  # 如果是二分类问题
    recall = recall_score(y_true, y_pred, average='binary')  # 如果是二分类问题
    f1 = f1_score(y_true, y_pred, average='binary')  # 如果是二分类问题

    valid_logger.write(f"Validation completed.\n\n")

    # 输出结果
    valid_logger.write(f"Accuracy: {accuracy:.4f}\n")
    valid_logger.write(f"Precision: {precision:.4f}\n")
    valid_logger.write(f"Recall: {recall:.4f}\n")
    valid_logger.write(f"F1-Score: {f1:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./heart_failure_clinical_records_dataset.csv")
    parser.add_argument("--model_path", type=str, default="./svm_model.joblib")
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--train_log_path", type=str, default="train_log.txt")
    parser.add_argument("--valid_log_path", type=str, default="valid_log.txt")
    parser.add_argument("--kernel_type", type=str, default="linear")
    parser.add_argument("--C", type=float, default=3.0)
    parser.add_argument("--epsilon", type=float, default=1e-6)

    args = parser.parse_args()

    # valid(args)
    train(args)