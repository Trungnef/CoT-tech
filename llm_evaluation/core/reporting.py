"""
Module để tạo báo cáo và trực quan hóa kết quả đánh giá LLM.
Sử dụng dữ liệu sau phân tích để tạo các báo cáo HTML, CSV, và các biểu đồ.
"""

import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import base64
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib import cm
import matplotlib.colors as mcolors

# Tạo thư mục gốc và các thư mục con
os.makedirs("logs", exist_ok=True)

# Cấu hình logging
logger = logging.getLogger("reporting")
logger.setLevel(logging.INFO)

# Kiểm tra xem đã có handler chưa để tránh duplicate logs
if not logger.handlers:
    # Tạo file handler
    file_handler = logging.FileHandler(f"logs/reporting.log", encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Tạo console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Tạo formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Thêm handlers vào logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Setting cho matplotlib
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.labelpad': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.formatter.useoffset': False
})
plt.style.use('seaborn-v0_8-whitegrid')

# Tạo bảng màu đẹp cho các biểu đồ
COLORS = [
    '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
    '#1abc9c', '#d35400', '#34495e', '#7f8c8d', '#2c3e50',
    '#27ae60', '#e67e22', '#8e44ad', '#16a085', '#f1c40f'
]

class Reporting:
    def __init__(self, results_df, output_dir, timestamp, language="vietnamese"):
        """
        Khởi tạo đối tượng Reporting để tạo báo cáo và trực quan hóa kết quả.
        
        Args:
            results_df (pd.DataFrame): DataFrame chứa kết quả đã phân tích
            output_dir (str): Thư mục đầu ra để lưu các báo cáo và biểu đồ
            timestamp (str): Timestamp sử dụng cho tên các file đầu ra
            language (str, optional): Ngôn ngữ của báo cáo. Mặc định là "vietnamese"
        """
        self.results_df = results_df
        self.output_dir = output_dir
        self.timestamp = timestamp
        self.language = language.lower()
        
        # Tạo các đường dẫn thư mục cần thiết
        self.reports_dir = os.path.join(output_dir, "reports")
        self.plots_dir = os.path.join(output_dir, "plots")
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        logger.info(f"Đã khởi tạo Reporting với {len(results_df)} kết quả")
        
    def generate_reports(self):
        """
        Tạo tất cả các báo cáo và visualizations.
        
        Returns:
            dict: Dictionary chứa đường dẫn tới các báo cáo đã tạo
        """
        logger.info("Bắt đầu tạo báo cáo...")
        report_paths = {}
        
        try:
            # Tạo visualizations
            plot_paths = self._generate_visualizations()
            
            # Tạo báo cáo HTML
            html_path = self._create_html_report(plot_paths)
            if html_path:
                report_paths['html'] = html_path
            
            # Tạo báo cáo CSV
            csv_path = self._create_csv_report()
            if csv_path:
                report_paths['csv'] = csv_path
            
            # Tạo báo cáo JSON
            json_path = self._create_json_report()
            if json_path:
                report_paths['json'] = json_path
            
            # Tạo báo cáo Markdown
            md_path = self._create_markdown_report(plot_paths)
            if md_path:
                report_paths['markdown'] = md_path
            
            logger.info(f"Đã tạo {len(report_paths)} báo cáo")
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo: {str(e)}")
            logger.debug(traceback.format_exc())
        
        return report_paths
    
    def _create_accuracy_by_model_plot(self):
        """
        Tạo biểu đồ accuracy theo model.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'is_correct' not in self.results_df.columns:
            return self._create_fallback_plot("Accuracy theo Model", "Không có dữ liệu độ chính xác (is_correct)")
        
        # Tính accuracy theo model
        accuracy_by_model = self.results_df.groupby('model_name')['is_correct'].mean().reset_index()
        
        # Sắp xếp theo độ chính xác giảm dần để dễ so sánh
        accuracy_by_model = accuracy_by_model.sort_values('is_correct', ascending=False)
        
        # Tạo biểu đồ
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Tạo biểu đồ cột với màu gradient
        palette = sns.color_palette("viridis", len(accuracy_by_model))
        bars = sns.barplot(x='model_name', y='is_correct', data=accuracy_by_model, palette=palette, ax=ax)
        
        # Thêm giá trị lên đầu mỗi cột với định dạng phần trăm
        for i, p in enumerate(bars.patches):
            percentage = f'{p.get_height():.1%}'
            ax.annotate(percentage, 
                       (p.get_x() + p.get_width() / 2., p.get_height() + 0.01), 
                       ha = 'center', va = 'bottom', 
                       fontsize=14, fontweight='bold',
                       color='#333333')
        
        # Thêm nhãn và tiêu đề
        ax.set_title('Độ chính xác (Accuracy) theo Model', fontsize=18, pad=20, fontweight='bold')
        ax.set_xlabel('Model', fontsize=15, labelpad=15)
        ax.set_ylabel('Accuracy', fontsize=15, labelpad=15)
        ax.set_ylim(0, min(1.1, max(accuracy_by_model['is_correct']) * 1.15))  # Tự động điều chỉnh khoảng
        
        # Cải thiện trục x
        plt.xticks(rotation=30, ha='right', fontsize=13)
        
        # Thêm lưới ngang cho dễ đọc
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)  # Đảm bảo lưới ở dưới thanh
        
        # Thêm nhãn phần trăm ở trục y
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Thêm đường viền nhẹ
        for spine in ax.spines.values():
            spine.set_edgecolor('#dddddd')
        
        # Thêm chú thích cuối biểu đồ
        plt.figtext(0.5, 0.01, 
                  f'Dựa trên {len(self.results_df)} kết quả đánh giá', 
                  ha='center', fontsize=12, fontstyle='italic')
        
        plt.tight_layout()
        
        # Lưu biểu đồ với độ phân giải cao
        output_path = os.path.join(self.plots_dir, f"accuracy_by_model_{self.timestamp}.png")
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_accuracy_by_prompt_plot(self):
        """
        Tạo biểu đồ accuracy theo loại prompt.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'is_correct' not in self.results_df.columns:
            return self._create_fallback_plot("Accuracy theo Prompt", "Không có dữ liệu độ chính xác (is_correct)")
        
        # Tính accuracy theo prompt
        accuracy_by_prompt = self.results_df.groupby('prompt_type')['is_correct'].mean().reset_index()
        
        # Tạo biểu đồ
        plt.figure(figsize=(14, 8))
        
        # Tạo biểu đồ cột
        bar_plot = sns.barplot(x='prompt_type', y='is_correct', data=accuracy_by_prompt)
        
        # Thêm giá trị lên đầu mỗi cột
        for p in bar_plot.patches:
            bar_plot.annotate(f'{p.get_height():.2f}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'bottom', fontsize=11)
        
        # Thêm nhãn và tiêu đề
        plt.title('Accuracy theo Loại Prompt', fontsize=16)
        plt.xlabel('Loại Prompt', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha='right')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Lưu biểu đồ
        output_path = os.path.join(self.plots_dir, f"accuracy_by_prompt_{self.timestamp}.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _create_accuracy_heatmap(self):
        """
        Tạo biểu đồ heatmap về accuracy theo model và prompt type.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'is_correct' not in self.results_df.columns:
            return self._create_fallback_plot("Accuracy Heatmap", "Không có dữ liệu độ chính xác (is_correct)")
        
        # Tính accuracy theo model và prompt
        heatmap_data = self.results_df.pivot_table(
            index='model_name', 
            columns='prompt_type', 
            values='is_correct',
            aggfunc='mean'
        )
        
        # Tạo biểu đồ
        plt.figure(figsize=(18, 12))
        
        # Tạo một colormap đẹp với chuyển màu rõ ràng
        custom_cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # Tạo heatmap với các cải tiến
        ax = sns.heatmap(
            heatmap_data, 
            annot=True, 
            cmap=custom_cmap,
            fmt='.1%',  # Hiển thị dưới dạng phần trăm
            linewidths=1,
            linecolor='white',
            vmin=0, 
            vmax=1,
            cbar_kws={'label': 'Accuracy Rate', 'shrink': 0.8}
        )
        
        # Thêm nhãn và tiêu đề
        plt.title('So sánh Accuracy theo Model và Loại Prompt', fontsize=20, pad=20, fontweight='bold')
        plt.xlabel('Loại Prompt', fontsize=16, labelpad=15)
        plt.ylabel('Model', fontsize=16, labelpad=15)
        
        # Cải thiện nhãn trục
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        
        # Thêm viền cho heatmap
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('gray')
            spine.set_linewidth(1)
        
        # Thêm chú thích về màu
        plt.figtext(0.5, 0.01, 
                  'Màu xanh đậm = accuracy cao, Màu trắng/vàng = accuracy thấp', 
                  ha='center', fontsize=12, fontstyle='italic')
        
        plt.tight_layout()
        
        # Lưu biểu đồ với độ phân giải cao
        output_path = os.path.join(self.plots_dir, f"accuracy_heatmap_{self.timestamp}.png")
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_simple_comparison_plot(self):
        """
        Tạo biểu đồ so sánh đơn giản giữa các mô hình.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'is_correct' not in self.results_df.columns:
            return self._create_fallback_plot("So sánh Model", "Không có dữ liệu độ chính xác (is_correct)")
        
        # Tạo biểu đồ
        plt.figure(figsize=(15, 10))
        
        # Tạo biểu đồ bar với hue là model_name
        g = sns.catplot(
            data=self.results_df,
            kind="bar",
            x="prompt_type",
            y="is_correct",
            hue="model_name",
            palette="deep",
            alpha=0.9,
            height=8,
            aspect=1.5,
            legend_out=False
        )
        
        # Thêm nhãn và tiêu đề
        g.set_xticklabels(rotation=45, ha="right")
        g.set(ylim=(0, 1))
        g.fig.suptitle('So sánh Accuracy giữa các mô hình', fontsize=16)
        plt.subplots_adjust(top=0.9)  # Để tránh chồng lấn giữa tiêu đề và biểu đồ
        
        # Thêm grid
        g.ax.grid(axis='y', linestyle='--', alpha=0.7)
        g.fig.tight_layout()
        
        # Lưu biểu đồ
        output_path = os.path.join(self.plots_dir, f"model_comparison_{self.timestamp}.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _create_html_report(self, plot_paths):
        """
        Tạo báo cáo HTML đầy đủ với các biểu đồ.
        
        Args:
            plot_paths: Dictionary chứa đường dẫn đến các biểu đồ
            
        Returns:
            str: Đường dẫn đến file báo cáo HTML
        """
        try:
            # Các chỉ số tổng quan
            total_questions = len(self.results_df)
            total_correct = len(self.results_df[self.results_df['is_correct'] == True])
            overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
            
            models = self.results_df['model_name'].unique()
            prompts = self.results_df['prompt_type'].unique()
            
            # Tạo HTML với thiết kế hiện đại hơn
            html = f"""
            <!DOCTYPE html>
            <html lang="vi">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Báo cáo Đánh giá LLM - {self.timestamp}</title>
                <style>
                    /* Thiết lập chung */
                    :root {{
                        --primary-color: #3498db;
                        --secondary-color: #2c3e50;
                        --success-color: #2ecc71;
                        --danger-color: #e74c3c;
                        --light-color: #ecf0f1;
                        --dark-color: #34495e;
                        --border-radius: 6px;
                        --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }}
                    
                    body {{ 
                        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; 
                        margin: 0; 
                        padding: 0; 
                        background-color: #f8f9fa; 
                        color: #333;
                        line-height: 1.6;
                    }}
                    
                    .container {{ 
                        max-width: 1400px; 
                        margin: 0 auto; 
                        background-color: white; 
                        padding: 30px; 
                        box-shadow: var(--box-shadow);
                        border-radius: var(--border-radius);
                        margin-top: 20px;
                        margin-bottom: 20px;
                    }}
                    
                    /* Tiêu đề */
                    .report-header {{
                        margin-bottom: 30px;
                        border-bottom: 2px solid var(--primary-color);
                        padding-bottom: 20px;
                    }}
                    
                    h1 {{ 
                        color: var(--secondary-color);
                        font-size: 2.5rem;
                        margin-bottom: 10px;
                    }}
                    
                    h2 {{ 
                        color: var(--primary-color); 
                        font-size: 1.8rem;
                        margin-top: 40px;
                        padding-bottom: 10px;
                        border-bottom: 1px solid #eee;
                    }}
                    
                    h3 {{ 
                        color: var(--dark-color);
                        font-size: 1.4rem;
                    }}
                    
                    .timestamp {{
                        font-style: italic;
                        color: #666;
                        font-size: 1rem;
                    }}
                    
                    /* Thẻ số liệu */
                    .stats-container {{
                        display: flex;
                        justify-content: space-between;
                        flex-wrap: wrap;
                        gap: 20px;
                        margin: 30px 0;
                    }}
                    
                    .stat-card {{
                        flex: 1;
                        min-width: 250px;
                        background: white;
                        padding: 20px;
                        border-radius: var(--border-radius);
                        box-shadow: var(--box-shadow);
                        text-align: center;
                        transition: transform 0.3s ease;
                        border-top: 4px solid var(--primary-color);
                    }}
                    
                    .stat-card:hover {{
                        transform: translateY(-5px);
                    }}
                    
                    .stat-card:nth-child(1) {{
                        border-top-color: #3498db;
                    }}
                    
                    .stat-card:nth-child(2) {{
                        border-top-color: #2ecc71;
                    }}
                    
                    .stat-card:nth-child(3) {{
                        border-top-color: #f39c12;
                    }}
                    
                    .stat-value {{
                        font-size: 2.5rem;
                        font-weight: bold;
                        margin: 15px 0;
                        color: var(--secondary-color);
                    }}
                    
                    .stat-label {{
                        font-size: 1.1rem;
                        color: #666;
                        font-weight: 500;
                    }}
                    
                    /* Danh sách mô hình và prompt */
                    .model-list, .prompt-list {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 15px;
                        margin: 20px 0;
                    }}
                    
                    .model-card, .prompt-card {{
                        padding: 15px;
                        border-radius: var(--border-radius);
                        background-color: white;
                        box-shadow: var(--box-shadow);
                        width: calc(33% - 20px);
                        min-width: 250px;
                    }}
                    
                    .model-card {{
                        border-left: 4px solid var(--primary-color);
                    }}
                    
                    .prompt-card {{
                        border-left: 4px solid var(--dark-color);
                    }}
                    
                    .model-name, .prompt-name {{
                        font-weight: bold;
                        font-size: 1.2rem;
                        margin-bottom: 10px;
                    }}
                    
                    .model-accuracy, .prompt-accuracy {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }}
                    
                    .accuracy-bar {{
                        height: 8px;
                        background-color: #e9ecef;
                        border-radius: 4px;
                        width: 70%;
                        overflow: hidden;
                    }}
                    
                    .accuracy-fill {{
                        height: 100%;
                        background-color: var(--success-color);
                    }}
                    
                    .accuracy-value {{
                        font-weight: bold;
                        color: var(--success-color);
                    }}
                    
                    /* Biểu đồ */
                    .plots-container {{
                        margin: 40px 0;
                    }}
                    
                    .plot-card {{
                        background: white;
                        border-radius: var(--border-radius);
                        box-shadow: var(--box-shadow);
                        margin-bottom: 30px;
                        overflow: hidden;
                    }}
                    
                    .plot-title {{
                        background-color: var(--light-color);
                        padding: 15px 20px;
                        border-bottom: 1px solid #ddd;
                        font-size: 1.3rem;
                        font-weight: 600;
                    }}
                    
                    .plot-img-container {{
                        padding: 20px;
                        text-align: center;
                    }}
                    
                    .plot-img {{
                        max-width: 100%;
                        height: auto;
                        border-radius: 4px;
                    }}
                    
                    /* Bảng kết quả chi tiết */
                    .results-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 30px 0;
                        font-size: 14px;
                        box-shadow: var(--box-shadow);
                        border-radius: var(--border-radius);
                        overflow: hidden;
                    }}
                    
                    .results-table th, .results-table td {{
                        padding: 12px 15px;
                        border: 1px solid #ddd;
                        text-align: left;
                    }}
                    
                    .results-table th {{
                        background-color: var(--primary-color);
                        color: white;
                        font-weight: 600;
                        position: sticky;
                        top: 0;
                    }}
                    
                    .results-table tbody tr:nth-child(even) {{
                        background-color: #f8f9fa;
                    }}
                    
                    .results-table tbody tr:hover {{
                        background-color: #e9ecef;
                    }}
                    
                    .results-table .correct {{
                        color: var(--success-color);
                        font-weight: bold;
                    }}
                    
                    .results-table .incorrect {{
                        color: var(--danger-color);
                        font-weight: bold;
                    }}
                    
                    .result-text {{
                        max-height: 200px;
                        overflow-y: auto;
                        white-space: pre-wrap;
                        background-color: #f8f9fa;
                        border: 1px solid #eee;
                        padding: 10px;
                        border-radius: 4px;
                        font-family: 'Consolas', 'Courier New', monospace;
                        font-size: 13px;
                    }}
                    
                    /* Pagination for results table */
                    .pagination {{
                        display: flex;
                        justify-content: center;
                        margin: 20px 0;
                    }}
                    
                    .pagination button {{
                        background-color: var(--light-color);
                        border: none;
                        padding: 8px 16px;
                        margin: 0 5px;
                        cursor: pointer;
                        border-radius: 4px;
                        font-weight: bold;
                    }}
                    
                    .pagination button:hover, .pagination button.active {{
                        background-color: var(--primary-color);
                        color: white;
                    }}
                    
                    /* Footer */
                    .report-footer {{
                        margin-top: 40px;
                        padding-top: 20px;
                        border-top: 1px solid #eee;
                        text-align: center;
                        color: #666;
                    }}
                    
                    /* Collapse/Expand for large sections */
                    .collapsible-header {{
                        background-color: var(--light-color);
                        padding: 10px 15px;
                        border-radius: 4px;
                        cursor: pointer;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 10px;
                    }}
                    
                    .collapsible-header::after {{
                        content: '+';
                        font-size: 1.5rem;
                    }}
                    
                    .collapsible-header.active::after {{
                        content: '-';
                    }}
                    
                    .collapsible-content {{
                        display: none;
                        padding: 0 15px;
                    }}
                    
                    .collapsible-content.show {{
                        display: block;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="report-header">
                        <h1>Báo cáo Đánh giá LLM</h1>
                        <p class="timestamp">Thời gian tạo: {self.timestamp}</p>
                    </div>
                    
                    <h2>Tổng quan Kết quả</h2>
                    <div class="stats-container">
                        <div class="stat-card">
                            <div class="stat-label">Tổng số câu hỏi</div>
                            <div class="stat-value">{total_questions}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Câu trả lời đúng</div>
                            <div class="stat-value">{total_correct}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Accuracy tổng thể</div>
                            <div class="stat-value">{overall_accuracy:.1%}</div>
                        </div>
                    </div>
                    
                    <h2>Hiệu suất theo Model</h2>
                    <div class="model-list">
            """
            
            # Thêm thông tin về mỗi model với hiệu ứng thanh ngang
            for model in models:
                model_df = self.results_df[self.results_df['model_name'] == model]
                model_correct = len(model_df[model_df['is_correct'] == True])
                model_total = len(model_df)
                model_accuracy = model_correct / model_total if model_total > 0 else 0
                accuracy_percentage = model_accuracy * 100
                
                html += f"""
                <div class="model-card">
                    <div class="model-name">{model}</div>
                    <div>Câu đúng: {model_correct}/{model_total}</div>
                    <div class="model-accuracy">
                        <div class="accuracy-bar">
                            <div class="accuracy-fill" style="width: {accuracy_percentage}%;"></div>
                        </div>
                        <div class="accuracy-value">{model_accuracy:.1%}</div>
                    </div>
                </div>
                """
            
            html += """
                    </div>
                    
                    <h2>Hiệu suất theo Loại Prompt</h2>
                    <div class="prompt-list">
            """
            
            # Thêm thông tin về mỗi loại prompt
            for prompt in prompts:
                prompt_df = self.results_df[self.results_df['prompt_type'] == prompt]
                prompt_correct = len(prompt_df[prompt_df['is_correct'] == True])
                prompt_total = len(prompt_df)
                prompt_accuracy = prompt_correct / prompt_total if prompt_total > 0 else 0
                accuracy_percentage = prompt_accuracy * 100
                
                html += f"""
                <div class="prompt-card">
                    <div class="prompt-name">{prompt}</div>
                    <div>Câu đúng: {prompt_correct}/{prompt_total}</div>
                    <div class="prompt-accuracy">
                        <div class="accuracy-bar">
                            <div class="accuracy-fill" style="width: {accuracy_percentage}%;"></div>
                        </div>
                        <div class="accuracy-value">{prompt_accuracy:.1%}</div>
                    </div>
                </div>
                """
            
            html += """
                    </div>
                    
                    <h2>Các Biểu đồ Phân tích</h2>
                    <div class="plots-container">
            """
            
            # Thêm các biểu đồ với lớp bọc UI đẹp hơn
            for plot_name, plot_path in plot_paths.items():
                try:
                    # Đảm bảo đường dẫn tồn tại
                    if not os.path.exists(plot_path):
                        continue
                    
                    # Tạo tên hiển thị
                    display_name = " ".join(word.capitalize() for word in plot_name.split("_"))
                    
                    # Chuyển ảnh thành dạng base64 để nhúng vào HTML
                    with open(plot_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    html += f"""
                    <div class="plot-card">
                        <div class="plot-title">{display_name}</div>
                        <div class="plot-img-container">
                            <img class="plot-img" src="data:image/png;base64,{encoded_string}" alt="{display_name}">
                        </div>
                    </div>
                    """
                except Exception as e:
                    logger.error(f"Lỗi khi thêm biểu đồ {plot_name}: {str(e)}")
            
            # Thêm bảng kết quả chi tiết
            html += """
                    </div>
                    
                    <h2>Kết quả chi tiết</h2>
                    <div style="overflow-x:auto;">
                        <table class="results-table">
                            <thead>
                                <tr>
                                    <th>Mô hình</th>
                                    <th>Prompt</th>
                                    <th>Câu hỏi</th>
                                    <th>Câu trả lời</th>
                                    <th>Đáp án đúng</th>
                                    <th>Đánh giá</th>
                                </tr>
                            </thead>
                            <tbody>
            """
            
            # Lấy 20 mẫu ngẫu nhiên từ kết quả thay vì chỉ 10
            if len(self.results_df) > 20:
                sample_df = self.results_df.sample(20, random_state=42)
            else:
                sample_df = self.results_df
                
            for _, row in sample_df.iterrows():
                is_correct = "Đúng" if row.get('is_correct') == True else "Sai"
                accuracy_class = "correct" if row.get('is_correct') == True else "incorrect"
                
                # Lấy và chuẩn bị nội dung
                question = row.get('question_text', '')
                model_answer = row.get('response', '')
                correct_answer = row.get('correct_answer', '')
                
                # Hiển thị đầy đủ nội dung không giới hạn
                # Và dùng hàm định dạng chuỗi Python để tránh lỗi HTML
                html += f"""
                <tr>
                    <td>{row.get('model_name', '')}</td>
                    <td>{row.get('prompt_type', '')}</td>
                    <td>
                        <div class="result-text">{question}</div>
                    </td>
                    <td>
                        <div class="result-text">{model_answer}</div>
                    </td>
                    <td>
                        <div class="result-text">{correct_answer}</div>
                    </td>
                    <td><span class="{accuracy_class}">{is_correct}</span></td>
                </tr>
                """
            
            # Thêm footer và JS để hỗ trợ interactive features
            html += """
                            </tbody>
                        </table>
                    </div>
                    <p>(Hiển thị một số kết quả ngẫu nhiên từ tập dữ liệu)</p>
                    
                    <div class="report-footer">
                        <p>Báo cáo được tạo tự động bởi hệ thống đánh giá LLM</p>
                    </div>
                </div>
                
                <script>
                    // JavaScript cho các chức năng tương tác
                    document.addEventListener('DOMContentLoaded', function() {
                        // Xử lý collapsible sections
                        var collHeaders = document.querySelectorAll('.collapsible-header');
                        
                        collHeaders.forEach(function(header) {
                            header.addEventListener('click', function() {
                                this.classList.toggle('active');
                                var content = this.nextElementSibling;
                                if (content.style.display === 'block') {
                                    content.style.display = 'none';
                                } else {
                                    content.style.display = 'block';
                                }
                            });
                        });
                    });
                </script>
            </body>
            </html>
            """
            
            # Lưu file HTML
            report_path = os.path.join(self.reports_dir, f"report_{self.timestamp}.html")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            logger.info(f"Đã tạo báo cáo HTML tại: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo HTML: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    def _create_csv_report(self):
        """
        Tạo báo cáo CSV.
        
        Returns:
            str: Đường dẫn đến file báo cáo CSV
        """
        try:
            # Lưu toàn bộ DataFrame
            report_path = os.path.join(self.reports_dir, f"report_{self.timestamp}.csv")
            self.results_df.to_csv(report_path, index=False)
            
            logger.info(f"Đã tạo báo cáo CSV tại: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo CSV: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    def _create_json_report(self):
        """
        Tạo báo cáo JSON.
        
        Returns:
            str: Đường dẫn đến file báo cáo JSON
        """
        try:
            # Tạo JSON tổng quan
            models = self.results_df['model_name'].unique()
            prompts = self.results_df['prompt_type'].unique()
            
            report_data = {
                "meta": {
                    "timestamp": self.timestamp,
                    "total_questions": len(self.results_df),
                    "total_correct": len(self.results_df[self.results_df['is_correct'] == True]),
                },
                "models": {},
                "prompts": {}
            }
            
            # Thông tin về mỗi model
            for model in models:
                model_df = self.results_df[self.results_df['model_name'] == model]
                model_correct = len(model_df[model_df['is_correct'] == True])
                model_total = len(model_df)
                
                report_data["models"][model] = {
                    "correct": int(model_correct),
                    "total": int(model_total),
                    "accuracy": float(model_correct / model_total if model_total > 0 else 0)
                }
            
            # Thông tin về mỗi loại prompt
            for prompt in prompts:
                prompt_df = self.results_df[self.results_df['prompt_type'] == prompt]
                prompt_correct = len(prompt_df[prompt_df['is_correct'] == True])
                prompt_total = len(prompt_df)
                
                report_data["prompts"][prompt] = {
                    "correct": int(prompt_correct),
                    "total": int(prompt_total),
                    "accuracy": float(prompt_correct / prompt_total if prompt_total > 0 else 0)
                }
            
            # Lưu file JSON
            report_path = os.path.join(self.reports_dir, f"report_{self.timestamp}.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Đã tạo báo cáo JSON tại: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo JSON: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    def _create_markdown_report(self, plot_paths):
        """
        Tạo báo cáo Markdown.
        
        Args:
            plot_paths: Dictionary chứa đường dẫn đến các biểu đồ
            
        Returns:
            str: Đường dẫn đến file báo cáo Markdown
        """
        try:
            # Các chỉ số tổng quan
            total_questions = len(self.results_df)
            total_correct = len(self.results_df[self.results_df['is_correct'] == True])
            overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
            
            models = self.results_df['model_name'].unique()
            prompts = self.results_df['prompt_type'].unique()
            
            # Tạo Markdown
            md = f"""# Báo cáo Đánh giá LLM

Thời gian tạo: {self.timestamp}

## Tổng quan

- **Tổng số câu hỏi**: {total_questions}
- **Câu trả lời đúng**: {total_correct}
- **Accuracy tổng thể**: {overall_accuracy:.2%}

## Mô hình đã đánh giá

"""
            
            # Thêm thông tin về mỗi model
            for model in models:
                model_df = self.results_df[self.results_df['model_name'] == model]
                model_correct = len(model_df[model_df['is_correct'] == True])
                model_total = len(model_df)
                model_accuracy = model_correct / model_total if model_total > 0 else 0
                
                md += f"- **{model}**: {model_correct}/{model_total} câu đúng (accuracy: {model_accuracy:.2%})\n"
            
            md += "\n## Loại Prompt đã đánh giá\n\n"
            
            # Thêm thông tin về mỗi loại prompt
            for prompt in prompts:
                prompt_df = self.results_df[self.results_df['prompt_type'] == prompt]
                prompt_correct = len(prompt_df[prompt_df['is_correct'] == True])
                prompt_total = len(prompt_df)
                prompt_accuracy = prompt_correct / prompt_total if prompt_total > 0 else 0
                
                md += f"- **{prompt}**: {prompt_correct}/{prompt_total} câu đúng (accuracy: {prompt_accuracy:.2%})\n"
            
            md += "\n## Biểu đồ\n\n"
            
            # Thêm thông tin về các biểu đồ
            for plot_name, plot_path in plot_paths.items():
                if os.path.exists(plot_path):
                    # Tạo tên hiển thị
                    display_name = " ".join(word.capitalize() for word in plot_name.split("_"))
                    md += f"### {display_name}\n\n"
                    
                    # Tạo đường dẫn tương đối
                    rel_path = os.path.relpath(plot_path, self.reports_dir)
                    md += f"![{display_name}]({rel_path})\n\n"
            
            # Lưu file Markdown
            report_path = os.path.join(self.reports_dir, f"report_{self.timestamp}.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(md)
            
            logger.info(f"Đã tạo báo cáo Markdown tại: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo Markdown: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def _create_difficulty_performance_plot(self) -> str:
        """
        Tạo biểu đồ đánh giá hiệu suất dựa trên độ khó.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'difficulty' not in self.results_df.columns or 'is_correct' not in self.results_df.columns:
            return self._create_fallback_plot("Hiệu suất theo độ khó", "Không có dữ liệu độ khó")
        
        # Đảm bảo cột difficulty có giá trị
        df_valid = self.results_df.dropna(subset=['difficulty', 'is_correct'])
        
        if len(df_valid) == 0:
            return self._create_fallback_plot("Hiệu suất theo độ khó", "Không có dữ liệu hợp lệ")
            
        # Tính toán accuracy theo độ khó
        accuracy_by_difficulty = df_valid.groupby('difficulty')['is_correct'].mean().reset_index()
        
        # Đảm bảo có độ khó để vẽ
        if len(accuracy_by_difficulty) <= 1:
            return self._create_fallback_plot("Hiệu suất theo độ khó", "Chỉ có một mức độ khó")
        
        # Sắp xếp theo thứ tự độ khó (giả định: Dễ, Trung bình, Khó)
        difficulty_order = ['Dễ', 'Trung bình', 'Khó']
        
        # Lọc và sắp xếp các độ khó có trong dữ liệu
        available_difficulties = [d for d in difficulty_order if d in accuracy_by_difficulty['difficulty'].values]
        
        # Nếu không có độ khó nào khớp với thứ tự mặc định, sử dụng thứ tự từ dữ liệu
        if not available_difficulties:
            available_difficulties = accuracy_by_difficulty['difficulty'].unique()
        
        # Lọc dữ liệu theo các độ khó có sẵn
        accuracy_by_difficulty = accuracy_by_difficulty[accuracy_by_difficulty['difficulty'].isin(available_difficulties)]
        
        # Tạo bảng cho việc vẽ biểu đồ, đảm bảo thứ tự đúng
        if all(d in difficulty_order for d in accuracy_by_difficulty['difficulty']):
            # Sử dụng CategoricalDtype để sắp xếp
            from pandas.api.types import CategoricalDtype
            cat_type = CategoricalDtype(categories=difficulty_order, ordered=True)
            accuracy_by_difficulty['difficulty'] = accuracy_by_difficulty['difficulty'].astype(cat_type)
            accuracy_by_difficulty = accuracy_by_difficulty.sort_values('difficulty')
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 8))
        
        # Tạo biểu đồ cột
        bar_plot = sns.barplot(x='difficulty', y='is_correct', data=accuracy_by_difficulty)
        
        # Thêm giá trị lên đầu mỗi cột
        for p in bar_plot.patches:
            bar_plot.annotate(f'{p.get_height():.2f}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'bottom', fontsize=11)
        
        # Thêm nhãn và tiêu đề
        plt.title('Hiệu suất dựa trên độ khó của câu hỏi', fontsize=16)
        plt.xlabel('Độ khó', fontsize=14)
        plt.ylabel('Độ chính xác (Accuracy)', fontsize=14)
        plt.ylim(0, 1.0)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Lưu biểu đồ
        output_path = os.path.join(self.plots_dir, f"difficulty_performance_{self.timestamp}.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
        
    def _create_criteria_evaluation_plot(self) -> str:
        """
        Tạo biểu đồ đánh giá theo các tiêu chí dựa trên reasoning scores.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        # Kiểm tra nếu không có dữ liệu reasoning nào
        has_reasoning_data = any(col.startswith('reasoning_') and col != 'reasoning_scores_str' for col in self.results_df.columns)
        if not has_reasoning_data:
            return self._create_fallback_plot("Đánh giá tiêu chí", "Không có dữ liệu reasoning")
        
        # Khởi tạo dữ liệu cho biểu đồ
        criteria_data = []
        
        # Các tiêu chí cần trích xuất
        criteria = {
            'accuracy': 'Độ chính xác',
            'reasoning': 'Suy luận hợp lý',
            'completeness': 'Tính đầy đủ', 
            'explanation': 'Giải thích rõ ràng',
            'cultural_context': 'Phù hợp ngữ cảnh'
        }
        
        # Thu thập điểm cho từng model và tiêu chí
        for model in self.results_df['model_name'].unique():
            model_df = self.results_df[self.results_df['model_name'] == model]
            
            for criterion_key, criterion_name in criteria.items():
                scores = []
                
                for _, row in model_df.iterrows():
                    # Ưu tiên sử dụng các cột đã được flatten
                    column_name = f'reasoning_{criterion_key}'
                    
                    if column_name in row and pd.notna(row[column_name]):
                        score = row[column_name]
                        if isinstance(score, (int, float)) and score > 0:
                            scores.append(score)
                    # Trường hợp cũ: reasoning_scores là dictionary
                    elif isinstance(row.get('reasoning_scores'), dict):
                        score = row['reasoning_scores'].get(criterion_key, 0)
                        if isinstance(score, (int, float)) and score > 0:
                            scores.append(score)
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    
                    criteria_data.append({
                        'model': model,
                        'criterion': criterion_name,
                        'score': avg_score
                    })
        
        # Tạo DataFrame
        plot_df = pd.DataFrame(criteria_data)
        
        if len(plot_df) == 0:
            return self._create_fallback_plot("Đánh giá tiêu chí", "Không đủ dữ liệu để vẽ biểu đồ")
        
        # Vẽ biểu đồ
        plt.figure(figsize=(14, 10))
        
        # Tạo biểu đồ thanh
        bar_plot = sns.barplot(x='model', y='score', hue='criterion', data=plot_df)
        
        # Thêm nhãn và tiêu đề
        plt.title('Đánh giá các mô hình theo từng tiêu chí', fontsize=16)
        plt.xlabel('Mô hình', fontsize=14)
        plt.ylabel('Điểm đánh giá (1-5)', fontsize=14)
        plt.ylim(0, 5.5)
        
        # Thêm legend
        plt.legend(title='Tiêu chí', title_fontsize=12, fontsize=10, loc='best')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Lưu biểu đồ
        output_path = os.path.join(self.plots_dir, f"criteria_evaluation_{self.timestamp}.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _create_criteria_radar_plot(self) -> str:
        """
        Tạo biểu đồ radar hiển thị đánh giá theo các tiêu chí.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        # Kiểm tra nếu không có dữ liệu reasoning nào
        has_reasoning_data = any(col.startswith('reasoning_') and col != 'reasoning_scores_str' for col in self.results_df.columns)
        if not has_reasoning_data:
            return self._create_fallback_plot("Biểu đồ radar tiêu chí", "Không có dữ liệu đánh giá tiêu chí")
        
        # Lấy danh sách các model duy nhất
        models = self.results_df['model_name'].unique()
        
        # Các tiêu chí đánh giá
        criteria = ['accuracy', 'reasoning', 'completeness', 'explanation', 'cultural_context']
        criterion_labels = {
            'accuracy': 'Độ chính xác',
            'reasoning': 'Suy luận hợp lý',
            'completeness': 'Tính đầy đủ',
            'explanation': 'Giải thích rõ ràng',
            'cultural_context': 'Phù hợp ngữ cảnh'
        }
        
        # Tính điểm trung bình cho mỗi model và mỗi tiêu chí
        model_scores = {}
        for model in models:
            model_scores[model] = {}
            model_data = self.results_df[self.results_df['model_name'] == model]
            
            for criterion in criteria:
                scores = []
                column_name = f'reasoning_{criterion}'
                
                for _, row in model_data.iterrows():
                    # Ưu tiên sử dụng các cột đã được flatten
                    if column_name in row and pd.notna(row[column_name]):
                        score = row[column_name]
                        if isinstance(score, (int, float)) and score > 0:
                            scores.append(score)
                    # Trường hợp sử dụng reasoning_scores dạng dictionary
                    elif isinstance(row.get('reasoning_scores'), dict):
                        score = row['reasoning_scores'].get(criterion, 0)
                        if isinstance(score, (int, float)) and score > 0:
                            scores.append(score)
                
                model_scores[model][criterion] = sum(scores) / len(scores) if scores else 0
        
        # Tạo biểu đồ radar
        fig = plt.figure(figsize=(10, 10))
        
        # Số lượng biến
        N = len(criteria)
        
        # Góc của mỗi trục
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Đóng vòng tròn
        
        # Thiết lập subplot hình tròn
        ax = plt.subplot(111, polar=True)
        
        # Màu cho mỗi model
        colors = plt.cm.jet(np.linspace(0, 1, len(models)))
        
        # Vẽ biểu đồ radar cho mỗi model
        for i, model in enumerate(models):
            values = [model_scores[model].get(c, 0) for c in criteria]
            values += values[:1]  # Đóng vòng tròn
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=model)
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Thêm nhãn cho các tiêu chí
        criterion_labels_list = [criterion_labels.get(c, c) for c in criteria]
        plt.xticks(angles[:-1], criterion_labels_list)
        
        # Thiết lập giới hạn trục y
        ax.set_ylim(0, 5)
        
        # Thêm tiêu đề
        plt.title('Đánh giá các mô hình theo từng tiêu chí', size=15)
        
        # Thêm chú thích
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Lưu biểu đồ
        output_path = os.path.join(self.plots_dir, f"criteria_radar_{self.timestamp}.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
        
    def _create_context_adherence_plot(self) -> str:
        """
        Tạo biểu đồ so sánh độ phù hợp ngữ cảnh.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'prompt_type' not in self.results_df.columns or 'is_correct' not in self.results_df.columns:
            return self._create_fallback_plot("Đánh giá độ phù hợp ngữ cảnh", "Không có dữ liệu")
        
        # Lọc các loại prompt liên quan đến context và non-context
        context_prompts = ['few_shot_3', 'few_shot_5', 'few_shot_7', 'react', 'cot_few_shot_3', 'cot_few_shot_5', 'cot_few_shot_7']
        non_context_prompts = ['zero_shot', 'cot_zero_shot']
        
        df_context = self.results_df[self.results_df['prompt_type'].isin(context_prompts)]
        df_non_context = self.results_df[self.results_df['prompt_type'].isin(non_context_prompts)]
        
        if len(df_context) == 0 or len(df_non_context) == 0:
            return self._create_fallback_plot("Đánh giá độ phù hợp ngữ cảnh", "Không đủ dữ liệu để so sánh")
        
        # Tính accuracy theo model và loại prompt (context vs non-context)
        context_acc = df_context.groupby('model_name')['is_correct'].mean().reset_index()
        context_acc['type'] = 'Context-based'
        
        non_context_acc = df_non_context.groupby('model_name')['is_correct'].mean().reset_index()
        non_context_acc['type'] = 'Non-context'
        
        # Kết hợp dữ liệu
        combined_df = pd.concat([context_acc, non_context_acc])
        
        plt.figure(figsize=(12, 8))
        
        # Vẽ biểu đồ so sánh
        plot = sns.barplot(x='model_name', y='is_correct', hue='type', data=combined_df)
        
        # Thêm nhãn và tiêu đề
        plt.title('So sánh hiệu suất giữa Prompt có ngữ cảnh và không có ngữ cảnh', fontsize=15)
        plt.xlabel('Mô hình', fontsize=12)
        plt.ylabel('Độ chính xác (Accuracy)', fontsize=12)
        plt.ylim(0, 1.0)
        
        # Thêm giá trị cho mỗi cột
        for p in plot.patches:
            plot.annotate(f'{p.get_height():.2f}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        output_path = os.path.join(self.plots_dir, f"context_adherence_{self.timestamp}.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
        
    def _create_difficulty_comparison_plot(self) -> str:
        """
        Tạo biểu đồ so sánh hiệu suất theo độ khó cho từng mô hình.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'difficulty' not in self.results_df.columns or 'is_correct' not in self.results_df.columns:
            return self._create_fallback_plot("So sánh hiệu suất theo độ khó", "Không có dữ liệu độ khó")
        
        # Đảm bảo cột difficulty có giá trị
        df_valid = self.results_df.dropna(subset=['difficulty', 'is_correct'])
        
        if len(df_valid) == 0:
            return self._create_fallback_plot("So sánh hiệu suất theo độ khó", "Không có dữ liệu hợp lệ")
        
        # Lấy các mức độ khó duy nhất
        difficulty_levels = df_valid['difficulty'].unique()
        
        if len(difficulty_levels) <= 1:
            return self._create_fallback_plot("So sánh hiệu suất theo độ khó", "Chỉ có một mức độ khó")
        
        # Tính accuracy theo mô hình và độ khó
        model_difficulty_acc = df_valid.groupby(['model_name', 'difficulty'])['is_correct'].mean().reset_index()
        
        # Tạo biểu đồ line với marker
        plt.figure(figsize=(14, 8))
        
        # Vẽ biểu đồ line
        sns.pointplot(x='difficulty', y='is_correct', hue='model_name', data=model_difficulty_acc, markers=['o', 's', 'D', '^', 'v', '<', '>'], linestyles=['-', '--', '-.', ':', '-', '--', '-.'])
        
        # Thêm nhãn và tiêu đề
        plt.title('Hiệu suất của các mô hình theo độ khó của câu hỏi', fontsize=15)
        plt.xlabel('Độ khó', fontsize=12)
        plt.ylabel('Độ chính xác (Accuracy)', fontsize=12)
        plt.ylim(0, 1.0)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Lưu biểu đồ
        output_path = os.path.join(self.plots_dir, f"difficulty_comparison_{self.timestamp}.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _create_overall_criteria_comparison(self) -> str:
        """
        Tạo biểu đồ tổng quan so sánh tất cả các tiêu chí cho các mô hình.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        # Kiểm tra nếu không có dữ liệu reasoning nào
        has_reasoning_data = any(col.startswith('reasoning_') and col != 'reasoning_scores_str' for col in self.results_df.columns)
        if not has_reasoning_data:
            return self._create_fallback_plot("Tổng quan các tiêu chí", "Không có dữ liệu đánh giá")
        
        # Lấy danh sách các model duy nhất
        models = self.results_df['model_name'].unique()
        
        # Khởi tạo dữ liệu cho biểu đồ
        models_list = []
        criteria_list = []
        scores_list = []
        
        # Các tiêu chí đánh giá
        criteria = ['accuracy', 'reasoning', 'completeness', 'explanation', 'cultural_context']
        criterion_labels = {
            'accuracy': 'Độ chính xác',
            'reasoning': 'Suy luận hợp lý',
            'completeness': 'Tính đầy đủ',
            'explanation': 'Giải thích rõ ràng',
            'cultural_context': 'Phù hợp ngữ cảnh'
        }
        
        # Thu thập dữ liệu cho mỗi model và tiêu chí
        for model in models:
            model_data = self.results_df[self.results_df['model_name'] == model]
            
            for criterion in criteria:
                scores = []
                column_name = f'reasoning_{criterion}'
                
                for _, row in model_data.iterrows():
                    # Ưu tiên sử dụng các cột đã được flatten
                    if column_name in row and pd.notna(row[column_name]):
                        score = row[column_name]
                        if isinstance(score, (int, float)) and score > 0:
                            scores.append(score)
                    # Trường hợp sử dụng reasoning_scores dạng dictionary
                    elif isinstance(row.get('reasoning_scores'), dict):
                        score = row['reasoning_scores'].get(criterion, 0)
                        if isinstance(score, (int, float)) and score > 0:
                            scores.append(score)
                
                avg_score = sum(scores) / len(scores) if scores else 0
                
                models_list.append(model)
                criteria_list.append(criterion_labels.get(criterion, criterion))
                scores_list.append(avg_score)
        
        # Tạo DataFrame từ dữ liệu đã thu thập
        plot_df = pd.DataFrame({
            'model_name': models_list,
            'criterion': criteria_list,
            'score': scores_list
        })
        
        if len(plot_df) == 0:
            return self._create_fallback_plot("Tổng quan các tiêu chí", "Không đủ dữ liệu để vẽ biểu đồ")
        
        # Tạo biểu đồ thanh nhóm
        plt.figure(figsize=(16, 10))
        
        # Vẽ biểu đồ thanh nhóm
        sns.barplot(x='model_name', y='score', hue='criterion', data=plot_df)
        
        # Thêm nhãn và tiêu đề
        plt.title('Tổng quan đánh giá các mô hình theo tất cả tiêu chí', fontsize=16)
        plt.xlabel('Mô hình', fontsize=14)
        plt.ylabel('Điểm đánh giá (1-5)', fontsize=14)
        plt.ylim(0, 5.5)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Lưu biểu đồ
        output_path = os.path.join(self.plots_dir, f"overall_criteria_{self.timestamp}.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path

    def _create_fallback_plot(self, title: str, message: str = "Không đủ dữ liệu") -> str:
        """
        Tạo biểu đồ thay thế khi không đủ dữ liệu hoặc có lỗi.
        
        Args:
            title (str): Tiêu đề của biểu đồ
            message (str): Thông báo hiển thị trên biểu đồ
            
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        plt.figure(figsize=(10, 6))
        
        # Tạo biểu đồ trống với thông báo
        plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)
        plt.title(title, fontsize=16)
        plt.grid(False)
        plt.axis('off')
        
        # Xác định tên file
        sanitized_title = title.lower().replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
        output_path = os.path.join(self.plots_dir, f"fallback_{sanitized_title}_{self.timestamp}.png")
        
        # Lưu biểu đồ
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Đã tạo biểu đồ fallback '{title}': {message}")
        
        return output_path

    def _create_reasoning_criteria_plot(self):
        """Tạo biểu đồ radar về các tiêu chí đánh giá reasoning."""
        # Kiểm tra có đủ dữ liệu
        required_columns = ['model_name', 'reasoning_accuracy', 'reasoning_reasoning', 
                            'reasoning_completeness', 'reasoning_explanation', 
                            'reasoning_cultural_context', 'reasoning_average']
                            
        for col in required_columns:
            if col not in self.results_df.columns:
                return self._create_fallback_plot("Đánh giá tiêu chí", f"Thiếu cột dữ liệu: {col}")
        
        # Lọc các hàng có dữ liệu reasoning
        valid_df = self.results_df[~self.results_df['reasoning_average'].isna()]
        if len(valid_df) == 0:
            return self._create_fallback_plot("Đánh giá tiêu chí", "Không có dữ liệu reasoning")
        
        # Chuẩn bị dữ liệu
        criteria = ['accuracy', 'reasoning', 'completeness', 'explanation', 'cultural_context']
        model_scores = {}
        
        # Tính điểm trung bình cho mỗi model và tiêu chí
        for model in valid_df['model_name'].unique():
            model_df = valid_df[valid_df['model_name'] == model]
            scores = []
            
            for criterion in criteria:
                criterion_key = f'reasoning_{criterion}'
                avg_score = model_df[criterion_key].mean()
                scores.append(avg_score)
            
            model_scores[model] = scores
        
        # Tạo biểu đồ radar
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Thiết lập các góc cho biểu đồ radar
        angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False).tolist()
        angles += angles[:1]  # Khép vòng tròn
        
        # Vẽ biểu đồ cho từng model
        for i, (model, scores) in enumerate(model_scores.items()):
            scores = scores + scores[:1]  # Khép vòng tròn
            ax.plot(angles, scores, linewidth=2, label=model)
            ax.fill(angles, scores, alpha=0.1)
        
        # Thiết lập trục và nhãn
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'])
        ax.set_ylim(0, 5)
        
        plt.title('Đánh giá các tiêu chí lập luận', size=15)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Lưu biểu đồ
        plot_path = os.path.join(self.plots_dir, 'reasoning_criteria_plot.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _create_reasoning_by_prompt_plot(self):
        """Tạo biểu đồ so sánh chất lượng reasoning theo prompt type."""
        # Kiểm tra có đủ dữ liệu
        required_columns = ['model_name', 'prompt_type', 'reasoning_average']
        for col in required_columns:
            if col not in self.results_df.columns:
                return self._create_fallback_plot("Reasoning theo prompt", f"Thiếu cột dữ liệu: {col}")
        
        # Lọc các hàng có dữ liệu reasoning
        valid_df = self.results_df[~self.results_df['reasoning_average'].isna()]
        if len(valid_df) == 0:
            return self._create_fallback_plot("Reasoning theo prompt", "Không có dữ liệu reasoning")
        
        # Tính điểm trung bình cho mỗi cặp (model, prompt_type)
        pivot_df = valid_df.pivot_table(
            index='model_name', 
            columns='prompt_type', 
            values='reasoning_average',
            aggfunc='mean'
        ).fillna(0)
        
        # Tạo heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', vmin=0, vmax=5, 
                   linewidths=.5, fmt='.2f')
        
        plt.title('Chất lượng lập luận theo loại prompt', size=15)
        plt.tight_layout()
        
        # Lưu biểu đồ
        plot_path = os.path.join(self.plots_dir, 'reasoning_by_prompt_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _create_reasoning_by_question_type_plot(self):
        """Tạo biểu đồ so sánh chất lượng reasoning theo question type."""
        # Kiểm tra có đủ dữ liệu
        required_columns = ['model_name', 'question_type', 'reasoning_average']
        for col in required_columns:
            if col not in self.results_df.columns:
                return self._create_fallback_plot("Reasoning theo question type", f"Thiếu cột dữ liệu: {col}")
        
        # Lọc các hàng có dữ liệu reasoning
        valid_df = self.results_df[~self.results_df['reasoning_average'].isna()]
        if len(valid_df) == 0:
            return self._create_fallback_plot("Reasoning theo question type", "Không có dữ liệu reasoning")
        
        # Tính điểm trung bình cho mỗi cặp (model, question_type)
        grouped_df = valid_df.groupby(['model_name', 'question_type'])['reasoning_average'].mean().reset_index()
        
        # Tạo biểu đồ cột
        plt.figure(figsize=(14, 10))
        g = sns.catplot(
            data=grouped_df,
            kind="bar",
            x="question_type",
            y="reasoning_average",
            hue="model_name",
            palette="deep",
            height=6,
            aspect=1.5
        )
        
        g.set_xticklabels(rotation=45, ha="right")
        g.set(ylim=(0, 5))
        g.fig.suptitle('Chất lượng lập luận theo loại câu hỏi', fontsize=15)
        g.fig.tight_layout()
        
        # Lưu biểu đồ
        plot_path = os.path.join(self.plots_dir, 'reasoning_by_question_type_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path

    def _generate_visualizations(self) -> Dict[str, str]:
        """
        Tạo các biểu đồ visualization từ dữ liệu đánh giá.
        
        Returns:
            Dict[str, str]: Dictionary chứa đường dẫn đến các file biểu đồ
        """
        logger.info("Bắt đầu tạo các biểu đồ visualization")
        plot_paths = {}
        
        try:
            # Một số biểu đồ cơ bản
            if 'is_correct' in self.results_df.columns:
                # 1. Accuracy by Model
                plot_path = self._create_accuracy_by_model_plot()
                if plot_path:
                    plot_paths['accuracy_by_model'] = plot_path
                    
                # 2. Accuracy by Prompt Type 
                plot_path = self._create_accuracy_by_prompt_plot()
                if plot_path:
                    plot_paths['accuracy_by_prompt'] = plot_path
                    
                # 3. Accuracy Heatmap (Model x Prompt)
                plot_path = self._create_accuracy_heatmap()
                if plot_path:
                    plot_paths['accuracy_heatmap'] = plot_path
                    
                # 4. Biểu đồ so sánh đơn giản
                plot_path = self._create_simple_comparison_plot()
                if plot_path:
                    plot_paths['simple_comparison'] = plot_path
            
            # 5. Biểu đồ đánh giá độ khó
            plot_path = self._create_difficulty_performance_plot()
            if plot_path:
                plot_paths['difficulty_performance'] = plot_path
                
            # 6. Biểu đồ đánh giá theo từng tiêu chí
            plot_path = self._create_criteria_evaluation_plot()
            if plot_path:
                plot_paths['criteria_evaluation'] = plot_path

            # 7. Biểu đồ Radar các tiêu chí
            plot_path = self._create_criteria_radar_plot()
            if plot_path:
                plot_paths['criteria_radar'] = plot_path
                
            # 8. Biểu đồ so sánh độ phù hợp ngữ cảnh
            plot_path = self._create_context_adherence_plot()
            if plot_path:
                plot_paths['context_adherence'] = plot_path
                
            # 9. Biểu đồ so sánh hiệu suất theo độ khó
            plot_path = self._create_difficulty_comparison_plot()
            if plot_path:
                plot_paths['difficulty_comparison'] = plot_path
                
            # 10. Tạo biểu đồ tổng quan tất cả tiêu chí
            plot_path = self._create_overall_criteria_comparison()
            if plot_path:
                plot_paths['overall_criteria'] = plot_path
            
            # 11. Biểu đồ radar cho các tiêu chí đánh giá suy luận
            plot_path = self._create_reasoning_criteria_plot()
            if plot_path:
                plot_paths['reasoning_criteria'] = plot_path
            
            # 12. Biểu đồ heatmap cho chất lượng suy luận theo từng loại prompt
            plot_path = self._create_reasoning_by_prompt_plot()
            if plot_path:
                plot_paths['reasoning_by_prompt'] = plot_path
            
            # 13. Biểu đồ so sánh chất lượng suy luận theo từng loại câu hỏi
            plot_path = self._create_reasoning_by_question_type_plot()
            if plot_path:
                plot_paths['reasoning_by_question_type'] = plot_path
            
            logger.info(f"Đã tạo {len(plot_paths)} biểu đồ visualization")
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ: {str(e)}")
            logger.debug(traceback.format_exc())
        
        return plot_paths
