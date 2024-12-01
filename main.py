import os
import argparse
import numpy as np
import joblib

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
    model = SVM()

    # 训练模型
    support_vectors, iter = model.fit(X, Y)
    sv_count = support_vectors.shape[0]

    save_mode(args.model_path, model)

    # 保存日志
    if(args.log_dir is not None and not os.path.exists(args.log_dir)):
        os.mkdir(args.log_dir)
    train_logger = Logger(os.path.join(args.log_dir, args.train_log_path))

    train_logger.write(f"Train info\n-----------------\n")
    train_logger.write(f"max_iter: {model.max_iter}\n")
    train_logger.write(f"kernel_type: {model.kernel_type}\n")
    train_logger.write(f"C: {model.C}\n")
    train_logger.write(f"epsilon: {model.epsilon}\n\n")

    train_logger.write(f"Training completed.\n\n")

    train_logger.write(f"Support vector count: {sv_count}\n")
    train_logger.write(f"bias: {model.b:.3f}\n")
    train_logger.write(f"w: {model.w}\n")
    train_logger.write(f"Converged after {iter} iterations\n")
    train_logger.write(f"SVM model saved to {args.model_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./heart_failure_clinical_records_dataset.csv")
    parser.add_argument("--model_path", type=str, default="./svm_model.joblib")
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--train_log_path", type=str, default="train_log.txt")

    args = parser.parse_args()

    train(args)
