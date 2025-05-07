import tkinter as tk
from model import BananaMobileNetV3, load_datasets
from gui import BananaApp
import os

if __name__ == "__main__":
    # 检查并创建模型保存目录
    if not os.path.exists("mydata"):
        os.makedirs("mydata")
        os.makedirs("mydata/trainData")
        os.makedirs("mydata/testData")
        print("请将训练数据放入mydata/trainData，测试数据放入mydata/testData")

    root = tk.Tk()
    app = BananaApp(root)
    root.mainloop()