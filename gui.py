import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import threading
import torch

import model
from video_processor import VideoProcessor
import os
import cv2
from model import load_datasets
from model import BananaMobileNetV3
from model import load_datasets
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class BananaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("香蕉新鲜度检测系统")
        self.root.geometry("900x700")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = []
        self.model = None
        self.predictions = []

        _, _, self.class_names = load_datasets()

        self.init_model()

        self.current_screen = None
        self.show_home_screen()

    def init_model(self):
        """初始化MobileNetV3模型"""

        self.model = BananaMobileNetV3(num_classes=len(self.class_names)).to(self.device)
        model_path = "mobile_model.pth"
        train_loader, _, _ = load_datasets()
        print("当前工作目录:", os.getcwd())  # 打印程序运行时的工作目录
        print("模型文件是否存在:", os.path.exists("mobile_model.pth"))
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            self.model.train_model(train_loader, epochs=10)
            messagebox.showinfo("提示", "模型训练完成！")


    def show_home_screen(self):
        if self.current_screen:
            self.current_screen.destroy()
        self.current_screen = HomeScreen(self.root, self)
        self.current_screen.pack(fill=tk.BOTH, expand=True)

    def show_image_screen(self):
        if self.current_screen:
            self.current_screen.destroy()
        self.current_screen = ImageScreen(self.root, self)
        self.current_screen.pack(fill=tk.BOTH, expand=True)

    def show_video_screen(self):
        """显示视频检测界面"""
        if self.current_screen:
            self.current_screen.destroy()
        self.current_screen = VideoScreen(self.root, self)
        self.current_screen.pack(fill=tk.BOTH, expand=True)

    def show_stats_screen(self):
        """显示统计界面"""
        if self.current_screen:
            self.current_screen.destroy()
        self.current_screen = StatsScreen(self.root, self)
        self.current_screen.pack(fill=tk.BOTH, expand=True)

    def clear_predictions(self):
        """清空预测结果"""
        self.predictions = []
        messagebox.showinfo("提示", "已重置检测数据")
        self.show_home_screen()


class HomeScreen(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.configure(bg="#f5f5f5")
        self.create_widgets()
    def create_widgets(self):
        # 标题
        tk.Label(
            self,
            text="香蕉新鲜度检测系统",
            font=("Microsoft YaHei", 22, "bold"),
            bg="#f5f5f5",
            fg="#333"
        ).pack(pady=(30, 20))

        # 按钮行
        button_frame = tk.Frame(self, bg="#f5f5f5")
        button_frame.pack(pady=(0, 15))

        buttons = [
            ("图片检测", self.app.show_image_screen),
            ("视频检测", self.app.show_video_screen),
            ("结果统计", self.app.show_stats_screen),
            ("重新开始", self.app.clear_predictions)
        ]

        for text, cmd in buttons:
            tk.Button(
                button_frame,
                text=text,
                command=cmd,
                width=12,
                height=1,
                font=("Microsoft YaHei", 12),
                bg="#5D9CEC",
                fg="white",
                relief=tk.FLAT,
                activebackground="#4A89DC"
            ).pack(side=tk.LEFT, padx=12, ipadx=10)

        # 背景区域
        bg_area = tk.Frame(self, height=400, bg="#f5f5f5")
        bg_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        try:
            bg_img = ImageTk.PhotoImage(file="./beijtu.png")
            img_label = tk.Label(bg_area, image=bg_img, bg="#f5f5f5")
            img_label.image = bg_img
            img_label.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            print(f"加载背景图失败: {e}")
            tk.Label(bg_area,
                     text="香蕉新鲜度分析图表",
                     bg="#FFF3E0",
                     font=("Microsoft YaHei", 16)).pack(fill=tk.BOTH, expand=True)

class ImageScreen(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        # 返回按钮
        back_frame = tk.Frame(self)
        back_frame.pack(fill=tk.X, pady=10)
        tk.Button(
            back_frame,
            text="返回首页",
            command=self.app.show_home_screen,
            font=("Microsoft YaHei", 10)
        ).pack(side=tk.LEFT, padx=10)

        # 选择图片按钮
        select_frame = tk.Frame(self)
        select_frame.pack(pady=10)
        tk.Button(
            select_frame,
            text="选择图片",
            command=self.select_image,
            width=15,
            font=("Microsoft YaHei", 12)
        ).pack()

        # 图片显示区域
        self.image_frame = tk.Frame(self)
        self.image_frame.pack(pady=20)

        # 结果标签
        self.result_label = tk.Label(
            self,
            text="等待检测...",
            font=("Microsoft YaHei", 14)
        )
        self.result_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")]
        )
        if not file_path:
            return

        try:
            # 加载并显示图片
            pil_image = Image.open(file_path)
            pil_image_resized = pil_image.resize((500, 500))
            img_tk = ImageTk.PhotoImage(pil_image_resized)

            # 清除旧图片
            for widget in self.image_frame.winfo_children():
                widget.destroy()

            # 显示新图片
            img_label = tk.Label(self.image_frame, image=img_tk)
            img_label.image = img_tk
            img_label.pack()

            # 预处理图片
            image_tensor = self.app.model.transform(pil_image).unsqueeze(0).to(self.app.device)

            # 进行预测
            self.app.model.eval()
            with torch.no_grad():
                output = self.app.model(image_tensor)
                _, predicted = torch.max(output, 1)
                pred_class = self.app.class_names[predicted.item()]

            # 显示结果
            self.result_label.config(
                text=f"检测结果: {pred_class}",
                fg="green" if pred_class == "新鲜" else "orange" if pred_class == "半新鲜" else "red"
            )

            # 保存结果用于统计
            self.app.predictions.append((img_tk, pred_class))

        except Exception as e:
            messagebox.showerror("错误", f"图片处理失败: {str(e)}")

class VideoScreen(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.is_processing = False
        self.current_video = None
        self.create_widgets()

    def create_widgets(self):
        # 返回按钮
        back_frame = tk.Frame(self)
        back_frame.pack(fill=tk.X, pady=10)
        tk.Button(
            back_frame,
            text="返回首页",
            command=self.app.show_home_screen,
            font=("Microsoft YaHei", 10)
        ).pack(side=tk.LEFT, padx=10)

        # 选择视频按钮
        select_frame = tk.Frame(self)
        select_frame.pack(pady=10)
        tk.Button(
            select_frame,
            text="选择视频",
            command=self.select_video,
            width=15,
            font=("Microsoft YaHei", 12)
        ).pack()

        # 视频显示区域
        self.video_frame = tk.Frame(self)
        self.video_frame.pack(pady=20)

        # 进度条
        self.progress_bar = ttk.Progressbar(
            self,
            orient="horizontal",
            length=400,
            mode="determinate"
        )
        self.progress_bar.pack(pady=10)

        # 状态标签
        self.status_label = tk.Label(
            self,
            text="等待视频...",
            font=("Microsoft YaHei", 12)
        )
        self.status_label.pack(pady=5)

    def select_video(self):
        if self.is_processing:
            messagebox.showwarning("警告", "正在处理视频，请稍候")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("视频文件", "*.mp4 *.avi"), ("所有文件", "*.*")]
        )
        if not file_path:
            return

        self.current_video = file_path
        self.process_video()

    def process_video(self):
        if not self.current_video:
            return

        self.is_processing = True
        self.video_processor = VideoProcessor(
            self.app.model,
            self.app.model.transform,
            self.app.class_names
        )

        # 清除旧内容
        for widget in self.video_frame.winfo_children():
            widget.destroy()

        # 创建视频显示画布
        self.video_canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.video_canvas.pack()

        def update_ui(frame, pred, progress):
            frame = cv2.resize(frame, (640, 480))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_canvas.delete("all")
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.video_canvas.image = imgtk

            self.progress_bar["value"] = progress
            self.status_label.config(
                text=f"处理中: {progress:.1f}% | 当前状态: {pred}",
                fg="green" if pred == "新鲜" else "orange" if pred == "半新鲜" else "red"
            )
            self.update()

        def on_complete(stats):
            self.is_processing = False
            if stats:
                self.show_video_stats(stats)

        threading.Thread(
            target=lambda: on_complete(
                self.video_processor.process_video(self.current_video, update_ui)
            ),
            daemon=True
        ).start()

    def show_video_stats(self, stats):
        stats_window = tk.Toplevel(self)
        stats_window.title("视频统计结果")
        stats_window.geometry("480x360")

        tk.Label(
            stats_window,
            text="香蕉新鲜度分布",
            font=("Microsoft YaHei", 16, "bold")
        ).pack(pady=10)

        for state, percent in stats["stats"].items():
            frame = tk.Frame(stats_window)
            frame.pack(fill=tk.X, padx=20, pady=5)

            tk.Label(
                frame,
                text=f"{state}:",
                width=10,
                anchor="w",
                font=("Microsoft YaHei", 12)
            ).pack(side=tk.LEFT)

            tk.Label(
                frame,
                text=percent,
                font=("Microsoft YaHei", 12, "bold"),
                fg="green" if state == "新鲜" else "orange" if state == "半新鲜" else "red"
            ).pack(side=tk.LEFT)

        tk.Label(
            stats_window,
            text=f"主要状态: {stats['dominant_state']}",
            font=("Microsoft YaHei", 14, "bold"),
            fg="green" if stats['dominant_state'] == "新鲜" else "orange" if stats['dominant_state'] == "半新鲜" else "red"
        ).pack(pady=15)

        tk.Button(
            stats_window,
            text="关闭",
            command=stats_window.destroy,
            width=10,
            font=("Microsoft YaHei", 12)
        ).pack(pady=10)

class StatsScreen(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        if not self.app.predictions:
            tk.Label(self, text="暂无检测数据", font=("Microsoft YaHei", 14)).pack(pady=50)
            return

        # 返回按钮
        back_frame = tk.Frame(self)
        back_frame.pack(fill=tk.X, pady=10)
        tk.Button(
            back_frame,
            text="返回首页",
            command=self.app.show_home_screen,
            font=("Microsoft YaHei", 10)
        ).pack(side=tk.LEFT, padx=10)

        # 统计标题
        tk.Label(
            self,
            text="检测结果统计",
            font=("Microsoft YaHei", 18, "bold")
        ).pack(pady=15)

        try:
            # 计算统计信息
            count = Counter(pred for _, pred in self.app.predictions)
            total = len(self.app.predictions)
            states = list(count.keys())

            colors = []
            for state in states:
                if state == "新鲜":
                    colors.append("green")
                elif state == "半新鲜":
                    colors.append("orange")
                else:
                    colors.append("red")

            # 显示统计图表
            fig, ax = plt.subplots(figsize=(6, 4))
            counts = [count.get(state, 0) for state in states]

            ax.bar(states, counts, color=colors)
            ax.set_title("香蕉新鲜度统计")
            ax.set_xlabel("状态")
            ax.set_ylabel("数量")

            chart_frame = tk.Frame(self)
            chart_frame.pack(pady=20)
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

            # 显示统计文本
            stats_frame = tk.Frame(self)
            stats_frame.pack(pady=10)

            for state in states:
                frame = tk.Frame(stats_frame)
                frame.pack(fill=tk.X, padx=50, pady=5)

                tk.Label(
                    frame,
                    text=f"{state}:",
                    width=8,
                    anchor="w",
                    font=("Microsoft YaHei", 12)
                ).pack(side=tk.LEFT)

                cnt = count.get(state, 0)
                percent = (cnt / total) * 100 if total > 0 else 0
                tk.Label(
                    frame,
                    text=f"{cnt}次 ({percent:.1f}%)",
                    font=("Microsoft YaHei", 12, "bold"),
                    fg=colors[states.index(state)]
                ).pack(side=tk.LEFT)

        except Exception as e:
            tk.Label(self, text=f"生成统计图失败: {str(e)}", fg="red").pack()