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
        Creates accuracy chart by model.
        
        Returns:
            str: Path to the chart file
        """
        if 'is_correct' not in self.results_df.columns:
            return self._create_fallback_plot("Accuracy by Model", "No accuracy data available (is_correct)")
        
        # Calculate accuracy by model
        accuracy_by_model = self.results_df.groupby('model_name')['is_correct'].mean().reset_index()
        
        # Sort by accuracy in descending order for easier comparison
        accuracy_by_model = accuracy_by_model.sort_values('is_correct', ascending=False)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create bar chart with gradient colors
        palette = sns.color_palette("viridis", len(accuracy_by_model))
        bars = sns.barplot(x='model_name', y='is_correct', hue='model_name', data=accuracy_by_model, palette=palette, ax=ax, legend=False)
        
        # Add percentage values at the top of each bar
        for i, p in enumerate(bars.patches):
            percentage = f'{p.get_height():.1%}'
            ax.annotate(percentage, 
                       (p.get_x() + p.get_width() / 2., p.get_height() + 0.01), 
                       ha = 'center', va = 'bottom', 
                       fontsize=14, fontweight='bold',
                       color='#333333')
        
        # Thêm nhãn và tiêu đề
        ax.set_title('Accuracy by Model', fontsize=18, pad=20, fontweight='bold')
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
                  f'Based on {len(self.results_df)} evaluation results', 
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
            return self._create_fallback_plot("Accuracy by Prompt", "No accuracy data available (is_correct)")
        
        # Tính accuracy theo prompt
        accuracy_by_prompt = self.results_df.groupby('prompt_type')['is_correct'].mean().reset_index()
        
        # Tạo biểu đồ
        plt.figure(figsize=(14, 8))
        
        # Tạo biểu đồ cột
        bar_plot = sns.barplot(x='prompt_type', y='is_correct', hue='prompt_type', data=accuracy_by_prompt, legend=False)
        
        # Thêm giá trị lên đầu mỗi cột
        for p in bar_plot.patches:
            bar_plot.annotate(f'{p.get_height():.2f}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'bottom', fontsize=11)
        
        # Thêm nhãn và tiêu đề
        plt.title('Accuracy by Prompt Type', fontsize=16)
        plt.xlabel('Prompt Type', fontsize=14)
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
        Tạo biểu đồ heatmap thể hiện accuracy theo model và prompt.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'is_correct' not in self.results_df.columns:
            return self._create_fallback_plot("Accuracy Heatmap", "No accuracy data available (is_correct)")
        
        # Tính accuracy theo model và prompt
        accuracy_heatmap = self.results_df.groupby(['model_name', 'prompt_type'])['is_correct'].mean().unstack()
        
        # Tạo biểu đồ
        plt.figure(figsize=(15, 10))
        
        # Tạo heatmap
        ax = sns.heatmap(
            accuracy_heatmap, 
            annot=True, 
            cmap="YlGnBu", 
            fmt=".2f", 
            linewidths=.5,
            vmin=0, 
            vmax=1.0,
            cbar_kws={'label': 'Accuracy'}
        )
        
        # Thêm nhãn và tiêu đề
        plt.title('Model-Prompt Accuracy Heatmap', fontsize=18, pad=20)
        plt.xlabel('Prompt Type', fontsize=14, labelpad=10)
        plt.ylabel('Model', fontsize=14, labelpad=10)
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        output_path = os.path.join(self.plots_dir, f"accuracy_heatmap_{self.timestamp}.png")
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
        Tạo báo cáo HTML dựa trên kết quả phân tích và các biểu đồ đã tạo.
        
        Args:
            plot_paths (Dict[str, str]): Đường dẫn đến các biểu đồ
        
        Returns:
            str: Đường dẫn đến file báo cáo HTML
        """
        try:
            # Các chỉ số tổng quan
            total_questions = len(self.results_df)
            total_correct = len(self.results_df[self.results_df['is_correct'] == True]) if 'is_correct' in self.results_df.columns else 0
            overall_accuracy = total_correct / total_questions if total_questions > 0 and 'is_correct' in self.results_df.columns else 0
            
            # Lấy danh sách models và prompts
            models = self.results_df['model_name'].unique() if 'model_name' in self.results_df.columns else []
            prompts = self.results_df['prompt_type'].unique() if 'prompt_type' in self.results_df.columns else []
            
            # Tạo HTML với thiết kế hiện đại
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
                    
                    .plot-description {{
                        padding: 10px 20px;
                        color: #666;
                        font-size: 0.95rem;
                        border-bottom: 1px solid #eee;
                        background-color: #fafafa;
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
                    
                    /* Nhóm biểu đồ */
                    .plot-group {{
                        margin-bottom: 40px;
                    }}
                    
                    .plot-group-header {{
                        background-color: var(--secondary-color);
                        color: white;
                        padding: 10px 15px;
                        border-radius: var(--border-radius);
                        margin-bottom: 20px;
                        font-size: 1.2rem;
                        font-weight: 500;
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
                    
                    /* Tooltip */
                    .tooltip {{
                        position: relative;
                        display: inline-block;
                        cursor: help;
                    }}
                    
                    .tooltip .tooltiptext {{
                        visibility: hidden;
                        width: 200px;
                        background-color: #555;
                        color: #fff;
                        text-align: center;
                        border-radius: 6px;
                        padding: 5px;
                        position: absolute;
                        z-index: 1;
                        bottom: 125%;
                        left: 50%;
                        margin-left: -100px;
                        opacity: 0;
                        transition: opacity 0.3s;
                    }}
                    
                    .tooltip:hover .tooltiptext {{
                        visibility: visible;
                        opacity: 1;
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
            """
            
            # Thêm thống kê về câu trả lời đúng và accuracy nếu có dữ liệu is_correct
            if 'is_correct' in self.results_df.columns:
                html += f"""
                        <div class="stat-card">
                            <div class="stat-label">Câu trả lời đúng</div>
                            <div class="stat-value">{total_correct}</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Accuracy tổng thể</div>
                            <div class="stat-value">{overall_accuracy:.1%}</div>
                        </div>
                """
            
            html += """
                    </div>
            """
            
            # Hiển thị thông tin về Model nếu có
            if len(models) > 0:
                html += """
                    <h2>Hiệu suất theo Model</h2>
                    <div class="model-list">
                """
                
                # Thêm thông tin về mỗi model
                for model in models:
                    model_df = self.results_df[self.results_df['model_name'] == model]
                    model_total = len(model_df)
                    
                    # Tính accuracy nếu có dữ liệu is_correct
                    if 'is_correct' in self.results_df.columns:
                        model_correct = len(model_df[model_df['is_correct'] == True])
                        model_accuracy = model_correct / model_total if model_total > 0 else 0
                        accuracy_percentage = model_accuracy * 100
                        
                        accuracy_info = f"""
                        <div>Câu đúng: {model_correct}/{model_total}</div>
                        <div class="model-accuracy">
                            <div class="accuracy-bar">
                                <div class="accuracy-fill" style="width: {accuracy_percentage}%;"></div>
                            </div>
                            <div class="accuracy-value">{model_accuracy:.1%}</div>
                        </div>
                        """
                    else:
                        accuracy_info = f"<div>Số mẫu: {model_total}</div>"
                    
                    html += f"""
                    <div class="model-card">
                        <div class="model-name">{model}</div>
                        {accuracy_info}
                    </div>
                    """
                
                html += """
                    </div>
                """
            
            # Hiển thị thông tin về Prompt nếu có
            if len(prompts) > 0:
                html += """
                    <h2>Hiệu suất theo Loại Prompt</h2>
                    <div class="prompt-list">
                """
                
                # Thêm thông tin về mỗi loại prompt
                for prompt in prompts:
                    prompt_df = self.results_df[self.results_df['prompt_type'] == prompt]
                    prompt_total = len(prompt_df)
                    
                    # Tính accuracy nếu có dữ liệu is_correct
                    if 'is_correct' in self.results_df.columns:
                        prompt_correct = len(prompt_df[prompt_df['is_correct'] == True])
                        prompt_accuracy = prompt_correct / prompt_total if prompt_total > 0 else 0
                        accuracy_percentage = prompt_accuracy * 100
                        
                        accuracy_info = f"""
                        <div>Câu đúng: {prompt_correct}/{prompt_total}</div>
                        <div class="prompt-accuracy">
                            <div class="accuracy-bar">
                                <div class="accuracy-fill" style="width: {accuracy_percentage}%;"></div>
                            </div>
                            <div class="accuracy-value">{prompt_accuracy:.1%}</div>
                        </div>
                        """
                    else:
                        accuracy_info = f"<div>Số mẫu: {prompt_total}</div>"
                    
                    html += f"""
                    <div class="prompt-card">
                        <div class="prompt-name">{prompt}</div>
                        {accuracy_info}
                    </div>
                    """
                
                html += """
                    </div>
                """
            
            # Thêm các biểu đồ phân tích theo nhóm
            html += """
                <h2>Các Biểu đồ Phân tích</h2>
            """
            
            # Định nghĩa các nhóm biểu đồ và mô tả
            plot_groups = {
                "accuracy": {
                    "title": "Biểu đồ Độ chính xác (Accuracy)",
                    "plots": {
                        "accuracy_by_model": "Hiển thị độ chính xác (accuracy) của từng mô hình, giúp so sánh hiệu suất tổng thể giữa các mô hình.",
                        "accuracy_by_prompt": "So sánh độ chính xác theo từng loại prompt, giúp xác định loại prompt hiệu quả nhất.",
                        "accuracy_heatmap": "Hiển thị chi tiết độ chính xác của mỗi mô hình trên từng loại prompt dưới dạng heatmap.",
                        "simple_comparison": "So sánh tổng quan giữa các mô hình trên các loại prompt khác nhau."
                    }
                },
                "difficulty": {
                    "title": "Biểu đồ Phân tích theo Độ khó",
                    "plots": {
                        "difficulty_performance": "Hiển thị hiệu suất của các mô hình theo độ khó của câu hỏi.",
                        "difficulty_comparison": "So sánh hiệu suất của từng mô hình trên các mức độ khó khác nhau."
                    }
                },
                "reasoning": {
                    "title": "Biểu đồ Phân tích Lập luận (Reasoning)",
                    "plots": {
                        "criteria_evaluation": "Đánh giá các mô hình theo từng tiêu chí cụ thể.",
                        "criteria_radar": "Hiển thị điểm số trên các tiêu chí dưới dạng biểu đồ radar cho tổng quan trực quan.",
                        "overall_criteria": "So sánh tất cả tiêu chí đánh giá giữa các mô hình.",
                        "reasoning_criteria": "Phân tích chi tiết về khả năng lập luận của từng mô hình.",
                        "reasoning_by_prompt": "Hiển thị chất lượng lập luận của mô hình theo từng loại prompt.",
                        "reasoning_by_question_type": "Phân tích chất lượng lập luận của mô hình theo từng loại câu hỏi."
                    }
                },
                "context": {
                    "title": "Biểu đồ Phân tích Ngữ cảnh",
                    "plots": {
                        "context_adherence": "So sánh hiệu suất khi sử dụng prompt có ngữ cảnh và không có ngữ cảnh."
                    }
                }
            }
            
            # Thêm các biểu đồ theo nhóm
            for group_key, group_info in plot_groups.items():
                group_title = group_info["title"]
                group_plots = group_info["plots"]
                
                # Kiểm tra xem có biểu đồ nào thuộc nhóm này không
                group_has_plots = any(plot_key in plot_paths for plot_key in group_plots.keys())
                
                if group_has_plots:
                    html += f"""
                    <div class="plot-group">
                        <div class="plot-group-header">{group_title}</div>
                        <div class="plots-container">
                    """
                    
                    # Thêm từng biểu đồ trong nhóm
                    for plot_key, plot_description in group_plots.items():
                        if plot_key in plot_paths:
                            plot_info = plot_paths[plot_key]
                            
                            # Kiểm tra cấu trúc của plot_info
                            if isinstance(plot_info, dict) and 'path' in plot_info:
                                plot_path = plot_info['path']
                                description = plot_info.get('description', plot_description)
                            else:
                                # Cấu trúc cũ - chuỗi đường dẫn trực tiếp
                                plot_path = plot_info
                                description = plot_description
                            
                            # Đảm bảo đường dẫn tồn tại
                            if not os.path.exists(plot_path):
                                continue
                            
                            # Tạo tên hiển thị
                            display_name = " ".join(word.capitalize() for word in plot_key.split("_"))
                            
                            # Chuyển ảnh thành dạng base64 để nhúng vào HTML
                            with open(plot_path, "rb") as image_file:
                                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                            
                            html += f"""
                            <div class="plot-card">
                                <div class="plot-title">{display_name}</div>
                                <div class="plot-description">{description}</div>
                                <div class="plot-img-container">
                                    <img class="plot-img" src="data:image/png;base64,{encoded_string}" alt="{display_name}">
                                </div>
                            </div>
                            """
                    
                    html += """
                        </div>
                    </div>
                    """
            
            # Thêm bảng kết quả chi tiết nếu có dữ liệu là_correct
            if 'is_correct' in self.results_df.columns:
                html += """
                    <h2>Kết quả chi tiết</h2>
                    <div class="collapsible-header">Xem chi tiết kết quả đánh giá</div>
                    <div class="collapsible-content">
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
                
                # Lấy 20 mẫu ngẫu nhiên từ kết quả
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
                
                html += """
                                </tbody>
                            </table>
                        </div>
                        <p>(Hiển thị một số kết quả ngẫu nhiên từ tập dữ liệu)</p>
                    </div>
                """
            
            # Thêm thông tin về các tiêu chí đánh giá nếu có
            criteria_columns = [col for col in self.results_df.columns if col.startswith('reasoning_') and col != 'reasoning_scores_str']
            if criteria_columns:
                html += """
                    <h2>Thông tin tiêu chí đánh giá</h2>
                    <div class="collapsible-header">Xem mô tả các tiêu chí đánh giá reasoning</div>
                    <div class="collapsible-content">
                        <table class="results-table">
                            <thead>
                                <tr>
                                    <th>Tiêu chí</th>
                                    <th>Mô tả</th>
                                    <th>Thang điểm</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Độ chính xác (Accuracy)</td>
                                    <td>Đánh giá mức độ chính xác của câu trả lời so với đáp án chuẩn.</td>
                                    <td>1-5 (1: Hoàn toàn sai, 5: Hoàn toàn đúng)</td>
                                </tr>
                                <tr>
                                    <td>Suy luận hợp lý (Reasoning)</td>
                                    <td>Đánh giá quá trình suy luận, lập luận dẫn đến câu trả lời có hợp lý không.</td>
                                    <td>1-5 (1: Không có lập luận, 5: Lập luận rất logic)</td>
                                </tr>
                                <tr>
                                    <td>Tính đầy đủ (Completeness)</td>
                                    <td>Đánh giá mức độ đầy đủ của câu trả lời, có bao quát hết nội dung của câu hỏi không.</td>
                                    <td>1-5 (1: Thiếu nhiều, 5: Rất đầy đủ)</td>
                                </tr>
                                <tr>
                                    <td>Giải thích rõ ràng (Explanation)</td>
                                    <td>Đánh giá mức độ rõ ràng, dễ hiểu của phần giải thích trong câu trả lời.</td>
                                    <td>1-5 (1: Rất khó hiểu, 5: Rất rõ ràng)</td>
                                </tr>
                                <tr>
                                    <td>Phù hợp ngữ cảnh (Cultural Context)</td>
                                    <td>Đánh giá mức độ phù hợp với ngữ cảnh văn hóa, ngôn ngữ trong câu trả lời.</td>
                                    <td>1-5 (1: Không phù hợp, 5: Rất phù hợp)</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                """
            
            # Thêm footer và JS
            html += """
                    <div class="report-footer">
                        <p>Báo cáo được tạo tự động bởi hệ thống đánh giá LLM</p>
                        <p>© Trune Evaluation System</p>
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
                
                # Thêm thông tin về F1, METEOR và BERT score nếu có
                model_metrics = []
                
                if 'f1_score' in self.results_df.columns:
                    f1_avg = model_df['f1_score'].mean() 
                    if not pd.isna(f1_avg):
                        model_metrics.append(f"F1 Score: {f1_avg:.4f}")
                
                if 'meteor_score' in self.results_df.columns:
                    meteor_avg = model_df['meteor_score'].mean()
                    if not pd.isna(meteor_avg):
                        model_metrics.append(f"METEOR: {meteor_avg:.4f}")
                
                if 'bert_score' in self.results_df.columns:
                    bert_avg = model_df['bert_score'].mean()
                    if not pd.isna(bert_avg):
                        model_metrics.append(f"BERT Score: {bert_avg:.4f}")
                
                metrics_text = f" ({', '.join(model_metrics)})" if model_metrics else ""
                
                md += f"- **{model}**: {model_correct}/{model_total} câu đúng (accuracy: {model_accuracy:.2%}){metrics_text}\n"
            
            md += "\n## Loại Prompt đã đánh giá\n\n"
            
            # Thêm thông tin về mỗi loại prompt
            for prompt in prompts:
                prompt_df = self.results_df[self.results_df['prompt_type'] == prompt]
                prompt_correct = len(prompt_df[prompt_df['is_correct'] == True])
                prompt_total = len(prompt_df)
                prompt_accuracy = prompt_correct / prompt_total if prompt_total > 0 else 0
                
                md += f"- **{prompt}**: {prompt_correct}/{prompt_total} câu đúng (accuracy: {prompt_accuracy:.2%})\n"
            
            # Thêm phần đánh giá text similarity metrics nếu có dữ liệu
            has_similarity_metrics = any(col in self.results_df.columns for col in ['f1_score', 'meteor_score', 'bert_score'])
            
            if has_similarity_metrics:
                md += "\n## Đánh giá độ tương đồng văn bản\n\n"
                md += "Các metrics đánh giá độ tương đồng giữa câu trả lời của mô hình và đáp án chuẩn:\n\n"
                
                metrics_table = "| Metric | Mô tả | Giá trị trung bình |\n| --- | --- | --- |\n"
                
                if 'f1_score' in self.results_df.columns:
                    f1_avg = self.results_df['f1_score'].mean() 
                    metrics_table += f"| F1_SCORE | Đánh giá độ tương đồng văn bản | {f1_avg:.4f} |\n"
                
                if 'meteor_score' in self.results_df.columns:
                    meteor_avg = self.results_df['meteor_score'].mean()
                    metrics_table += f"| METEOR_SCORE | Đánh giá chất lượng dịch thuật | {meteor_avg:.4f} |\n"
                
                if 'bert_score' in self.results_df.columns:
                    bert_avg = self.results_df['bert_score'].mean()
                    metrics_table += f"| BERT_SCORE | Đánh giá độ tương đồng ngữ nghĩa | {bert_avg:.4f} |\n"
                
                md += metrics_table + "\n"
            
            md += "\n## Biểu đồ\n\n"
            
            # Thêm thông tin về các biểu đồ
            for plot_name, plot_info in plot_paths.items():
                # Kiểm tra cấu trúc của plot_info
                if isinstance(plot_info, dict) and 'path' in plot_info:
                    plot_path = plot_info['path']
                    description = plot_info.get('description', '')
                else:
                    # Cấu trúc cũ - chuỗi đường dẫn trực tiếp
                    plot_path = plot_info
                    description = ''
                
                if os.path.exists(plot_path):
                    # Tạo tên hiển thị
                    display_name = " ".join(word.capitalize() for word in plot_name.split("_"))
                    md += f"### {display_name}\n\n"
                    
                    # Thêm mô tả chi tiết hơn cho các biểu đồ F1, METEOR và BERT
                    if plot_name == 'f1_score':
                        md += "F1_SCORE đánh giá độ tương đồng văn bản dựa trên sự trùng lặp từ ngữ giữa câu trả lời và đáp án chuẩn. Giá trị từ 0-1, càng cao càng tốt.\n\n"
                    elif plot_name == 'meteor_score':
                        md += "METEOR_SCORE là thước đo đánh giá chất lượng dịch thuật, tính cả khả năng khớp từ vựng, đồng nghĩa và cấu trúc. Giá trị từ 0-1, càng cao càng tốt.\n\n"
                    elif plot_name == 'bert_score':
                        md += "BERT_SCORE đánh giá độ tương đồng ngữ nghĩa sử dụng mô hình ngôn ngữ BERT, xét đến ngữ cảnh sâu hơn so với chỉ đếm từ. Giá trị từ 0-1, càng cao càng tốt.\n\n"
                    # Không thay thế mô tả gốc cho các biểu đồ khác
                    elif description:
                        md += f"{description}\n\n"
                        
                    # Tạo đường dẫn tương đối
                    rel_path = os.path.relpath(plot_path, self.reports_dir)
                    md += f"![{display_name}]({rel_path})\n\n"
            
            # Thêm bảng kết quả chi tiết
            if len(self.results_df) > 0:
                md += "\n## Kết quả chi tiết\n\n"
                
                # Tạo bảng kết quả với thêm cột F1, METEOR và BERT Score
                table_cols = ["Model", "Prompt", "Accuracy"]
                
                if 'f1_score' in self.results_df.columns:
                    table_cols.append("F1_SCORE")
                if 'meteor_score' in self.results_df.columns:
                    table_cols.append("METEOR_SCORE")
                if 'bert_score' in self.results_df.columns:
                    table_cols.append("BERT_SCORE")
                
                table_header = " | ".join(table_cols)
                table_separator = " | ".join(["---" for _ in table_cols])
                
                md += f"| {table_header} |\n| {table_separator} |\n"
                
                # Tính toán kết quả chi tiết theo model và prompt
                model_prompt_results = []
                
                for model in models:
                    for prompt in prompts:
                        subset = self.results_df[(self.results_df['model_name'] == model) & 
                                                (self.results_df['prompt_type'] == prompt)]
                        
                        if len(subset) > 0:
                            accuracy = subset['is_correct'].mean()
                            result = {
                                "Model": model,
                                "Prompt": prompt,
                                "Accuracy": f"{accuracy:.2%}"
                            }
                            
                            # Thêm các metrics đo lường nếu có
                            if 'f1_score' in self.results_df.columns:
                                f1_avg = subset['f1_score'].mean()
                                result["F1_SCORE"] = f"{f1_avg:.4f}" if not pd.isna(f1_avg) else "N/A"
                                
                            if 'meteor_score' in self.results_df.columns:
                                meteor_avg = subset['meteor_score'].mean()
                                result["METEOR_SCORE"] = f"{meteor_avg:.4f}" if not pd.isna(meteor_avg) else "N/A"
                                
                            if 'bert_score' in self.results_df.columns:
                                bert_avg = subset['bert_score'].mean()
                                result["BERT_SCORE"] = f"{bert_avg:.4f}" if not pd.isna(bert_avg) else "N/A"
                            
                            model_prompt_results.append(result)
                
                # Thêm dòng cho từng cặp model-prompt
                for result in model_prompt_results:
                    row_values = [result[col] for col in table_cols]
                    row_str = " | ".join(row_values)
                    md += f"| {row_str} |\n"
            
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
        Tạo biểu đồ hiệu suất theo độ khó của câu hỏi.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'difficulty' not in self.results_df.columns or 'is_correct' not in self.results_df.columns:
            return self._create_fallback_plot("Performance by Difficulty", "No difficulty data available")
        
        # Tạo biểu đồ
        plt.figure(figsize=(12, 8))
        
        # Tính toán accuracy theo độ khó
        difficulty_perf = self.results_df.groupby('difficulty')['is_correct'].mean().reset_index()
        counts = self.results_df.groupby('difficulty').size().reset_index(name='count')
        difficulty_perf = pd.merge(difficulty_perf, counts, on='difficulty')
        
        # Tạo bar plot (sửa để tránh FutureWarning, thêm hue và legend=False)
        ax = sns.barplot(x='difficulty', y='is_correct', hue='difficulty', data=difficulty_perf, palette='viridis', legend=False)
        
        # Thêm số lượng mẫu lên mỗi cột
        for i, row in enumerate(difficulty_perf.itertuples()):
            ax.text(i, row.is_correct + 0.02, f'n={row.count}', ha='center')
            ax.text(i, row.is_correct/2, f'{row.is_correct:.1%}', ha='center', color='white', fontweight='bold')
        
        # Thêm labels và title
        plt.title('Performance based on Question Difficulty', fontsize=16)
        plt.xlabel('Difficulty Level', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
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
        
        # Tạo biểu đồ thanh - để seaborn tự động xử lý legend
        bar_plot = sns.barplot(x='model', y='score', hue='criterion', data=plot_df)
        
        # Thiết lập legend
        plt.legend(title='Tiêu chí', title_fontsize=12, fontsize=10, loc='best')
        
        plt.title('Đánh giá các mô hình theo từng tiêu chí', fontsize=16)
        plt.xlabel('Mô hình', fontsize=14)
        plt.ylabel('Điểm đánh giá (1-5)', fontsize=14)
        plt.ylim(0, 5.5)
        
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
        Tạo biểu đồ so sánh hiệu suất các mô hình theo độ khó.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'difficulty' not in self.results_df.columns or 'is_correct' not in self.results_df.columns:
            return self._create_fallback_plot("Model Performance by Difficulty", "No difficulty data available")
        
        # Tạo biểu đồ
        plt.figure(figsize=(14, 9))
        
        # Tính toán accuracy theo model và độ khó
        model_difficulty_perf = self.results_df.groupby(['model_name', 'difficulty'])['is_correct'].mean().reset_index()
        
        # Pivot để dễ vẽ biểu đồ
        pivot_data = model_difficulty_perf.pivot(index='model_name', columns='difficulty', values='is_correct')
        
        # Vẽ heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.1%', cmap='viridis', 
                   vmin=0, vmax=1, linewidths=.5, cbar_kws={'label': 'Accuracy'})
        
        # Thêm tiêu đề và labels
        plt.title('Model Performance by Question Difficulty Level', fontsize=15)
        plt.xlabel('Difficulty Level', fontsize=13)
        plt.ylabel('Model', fontsize=13)
        
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

    def _create_fallback_plot(self, title: str, message: str = "Insufficient data") -> str:
        """
        Creates a fallback plot when there is not enough data.
        
        Args:
            title: Chart title
            message: Error message
            
        Returns:
            str: Path to the chart file
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)
        ax.set_title(title)
        ax.axis('off')
        
        # Create filename from title
        filename = title.lower().replace(' ', '_')
        output_path = os.path.join(self.plots_dir, f"{filename}_{self.timestamp}.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path

    def _create_reasoning_criteria_plot(self):
        """Tạo biểu đồ radar về các tiêu chí đánh giá reasoning."""
        # Kiểm tra dữ liệu với cách tiếp cận linh hoạt hơn
        # Danh sách tiêu chí cần kiểm tra
        criteria = ['accuracy', 'reasoning', 'completeness', 'explanation', 'cultural_context']
        criteria_cols = [f'reasoning_{c}' for c in criteria]
        
        # Kiểm tra nếu có ít nhất một cột tiêu chí reasoning
        has_reasoning_data = any(col in self.results_df.columns for col in criteria_cols)
        if not has_reasoning_data:
            return self._create_fallback_plot("Đánh giá tiêu chí", "Không có dữ liệu đánh giá reasoning")
        
        # Chuẩn bị DataFrame hợp lệ cho việc phân tích
        valid_df = self.results_df
        
        # Nếu có cột reasoning_average, sử dụng để lọc dữ liệu
        if 'reasoning_average' in self.results_df.columns:
            valid_df = self.results_df[~self.results_df['reasoning_average'].isna()]
            if len(valid_df) == 0:
                return self._create_fallback_plot("Đánh giá tiêu chí", "Không có dữ liệu reasoning_average")
        else:
            # Tạo cột reasoning_average nếu không có
            for col in criteria_cols:
                if col not in self.results_df.columns:
                    valid_df[col] = np.nan
            
            # Tính toán điểm trung bình từ các cột có sẵn
            valid_cols = [col for col in criteria_cols if col in self.results_df.columns]
            if valid_cols:
                valid_df['reasoning_average'] = valid_df[valid_cols].mean(axis=1, skipna=True)
            else:
                return self._create_fallback_plot("Đánh giá tiêu chí", "Không đủ dữ liệu reasoning")
            
            # Lọc các hàng có dữ liệu reasoning
            valid_df = valid_df[~valid_df['reasoning_average'].isna()]
            if len(valid_df) == 0:
                return self._create_fallback_plot("Đánh giá tiêu chí", "Không có dữ liệu reasoning hợp lệ")
        
        # Tính điểm trung bình cho mỗi model và tiêu chí
        model_scores = {}
        
        for model in valid_df['model_name'].unique():
            model_df = valid_df[valid_df['model_name'] == model]
            scores = []
            
            for criterion in criteria:
                criterion_key = f'reasoning_{criterion}'
                if criterion_key in model_df.columns:
                    # Loại bỏ các giá trị NaN trước khi tính trung bình
                    valid_scores = model_df[criterion_key].dropna()
                    if len(valid_scores) > 0:
                        avg_score = valid_scores.mean()
                    else:
                        avg_score = 0
                else:
                    avg_score = 0
                scores.append(avg_score)
            
            model_scores[model] = scores
        
        # Tạo biểu đồ radar
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Thiết lập các góc cho biểu đồ radar
        angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False).tolist()
        angles += angles[:1]  # Khép vòng tròn
        
        # Tạo bảng màu đẹp cho các model
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_scores)))
        
        # Vẽ biểu đồ cho từng model
        for i, (model, scores) in enumerate(model_scores.items()):
            scores = scores + scores[:1]  # Khép vòng tròn
            ax.plot(angles, scores, linewidth=2, label=model, color=colors[i])
            ax.fill(angles, scores, alpha=0.1, color=colors[i])
        
        # Thiết lập trục và nhãn
        ax.set_xticks(angles[:-1])
        # Dùng tên tiêu chí dễ đọc hơn
        criterion_labels = {
            'accuracy': 'Độ chính xác',
            'reasoning': 'Suy luận hợp lý',
            'completeness': 'Tính đầy đủ',
            'explanation': 'Giải thích rõ ràng',
            'cultural_context': 'Phù hợp ngữ cảnh'
        }
        ax.set_xticklabels([criterion_labels.get(c, c) for c in criteria])
        
        # Thiết lập đường tròn mức độ
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'])
        ax.set_ylim(0, 5)
        
        plt.title('Đánh giá các tiêu chí lập luận', size=15, fontweight='bold')
        
        # Đặt legend ở vị trí tốt hơn
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        
        # Lưu biểu đồ với timestamp để tránh ghi đè
        plot_path = os.path.join(self.plots_dir, f"reasoning_criteria_plot_{self.timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _create_reasoning_by_prompt_plot(self):
        """Tạo biểu đồ so sánh chất lượng reasoning theo prompt type."""
        # Kiểm tra dữ liệu với cách tiếp cận linh hoạt hơn
        criteria = ['accuracy', 'reasoning', 'completeness', 'explanation', 'cultural_context']
        criteria_cols = [f'reasoning_{c}' for c in criteria]
        
        # Kiểm tra nếu có ít nhất một cột tiêu chí reasoning
        has_reasoning_data = any(col in self.results_df.columns for col in criteria_cols)
        if not has_reasoning_data:
            return self._create_fallback_plot("Reasoning theo prompt", "Không có dữ liệu đánh giá reasoning")
            
        # Kiểm tra cột prompt_type
        if 'prompt_type' not in self.results_df.columns:
            return self._create_fallback_plot("Reasoning theo prompt", "Không có dữ liệu về loại prompt")
        
        # Chuẩn bị DataFrame hợp lệ cho việc phân tích
        valid_df = self.results_df
        
        # Nếu có cột reasoning_average, sử dụng để lọc dữ liệu
        if 'reasoning_average' in self.results_df.columns:
            valid_df = valid_df[~valid_df['reasoning_average'].isna()]
        else:
            # Tạo cột reasoning_average nếu không có
            valid_cols = [col for col in criteria_cols if col in self.results_df.columns]
            if valid_cols:
                valid_df['reasoning_average'] = valid_df[valid_cols].mean(axis=1, skipna=True)
            else:
                return self._create_fallback_plot("Reasoning theo prompt", "Không đủ dữ liệu reasoning")
            
            # Lọc các hàng có dữ liệu reasoning
            valid_df = valid_df[~valid_df['reasoning_average'].isna()]
        
        # Kiểm tra sau khi lọc dữ liệu
        if len(valid_df) == 0:
            return self._create_fallback_plot("Reasoning theo prompt", "Không có dữ liệu reasoning hợp lệ")
        
        # Kiểm tra số lượng dữ liệu cho từng mô hình và loại prompt
        pivot_counts = valid_df.pivot_table(
            index='model_name', 
            columns='prompt_type', 
            values='reasoning_average',
            aggfunc='count'
        ).fillna(0)
        
        # Nếu không có đủ dữ liệu để phân tích
        if pivot_counts.sum().sum() < 3:  # Yêu cầu tối thiểu 3 điểm dữ liệu
            return self._create_fallback_plot("Reasoning theo prompt", "Không đủ dữ liệu để phân tích")
        
        # Tính điểm trung bình cho mỗi cặp (model, prompt_type)
        pivot_df = valid_df.pivot_table(
            index='model_name', 
            columns='prompt_type', 
            values='reasoning_average',
            aggfunc='mean'
        ).fillna(0)
        
        # Tạo heatmap
        plt.figure(figsize=(12, 8))
        
        # Tùy chỉnh colormap để phù hợp với thang điểm 0-5
        cmap = sns.color_palette("YlGnBu", as_cmap=True)
        
        # Tạo heatmap với các thông số cải tiến
        ax = sns.heatmap(
            pivot_df, 
            annot=True, 
            cmap=cmap, 
            vmin=0, 
            vmax=5, 
            linewidths=.5, 
            fmt='.2f',
            cbar_kws={'label': 'Điểm đánh giá (1-5)', 'shrink': 0.8}
        )
        
        # Cải thiện trình bày
        plt.title('Chất lượng lập luận theo loại prompt', size=16, fontweight='bold', pad=20)
        plt.xlabel('Loại Prompt', fontsize=14, labelpad=10)
        plt.ylabel('Mô hình', fontsize=14, labelpad=10)
        
        # Cải thiện nhãn trục
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
        # Thêm viền cho heatmap
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('#dddddd')
            spine.set_linewidth(0.5)
        
        plt.tight_layout()
        
        # Lưu biểu đồ với timestamp và độ phân giải cao
        plot_path = os.path.join(self.plots_dir, f"reasoning_by_prompt_plot_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _create_reasoning_by_question_type_plot(self):
        """
        Tạo biểu đồ đánh giá reasoning theo loại câu hỏi.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'question_type' not in self.results_df.columns:
            return self._create_fallback_plot("Reasoning by Question Type", "No question type data available")
            
        # Kiểm tra xem có dữ liệu reasoning không
        reasoning_cols = [col for col in self.results_df.columns if col.startswith('reasoning_') 
                        and col not in ['reasoning_evaluation', 'reasoning_scores_str']]
        
        if not reasoning_cols:
            return self._create_fallback_plot("Reasoning by Question Type", "No reasoning data available")
        
        try:
            # Chuẩn bị dữ liệu
            question_types = self.results_df['question_type'].unique()
            criteria_labels = {
                'reasoning_accuracy': 'Accuracy',
                'reasoning_logic': 'Logic',
                'reasoning_consistency': 'Consistency',
                'reasoning_difficulty': 'Difficulty',
                'reasoning_context': 'Context',
                'reasoning_average': 'Average'
            }
            
            # Tính toán điểm trung bình cho mỗi loại câu hỏi và tiêu chí
            data_for_plot = []
            for q_type in question_types:
                type_df = self.results_df[self.results_df['question_type'] == q_type]
                
                for col in reasoning_cols:
                    if col in self.results_df.columns:
                        label = criteria_labels.get(col, col.replace('reasoning_', '').title())
                        avg_score = type_df[col].mean()
                        data_for_plot.append({
                            'Question Type': q_type,
                            'Criteria': label,
                            'Score': avg_score
                        })
            
            if not data_for_plot:
                return self._create_fallback_plot("Reasoning by Question Type", "Insufficient reasoning data")
            
            # Tạo DataFrame cho dễ vẽ biểu đồ
            plot_df = pd.DataFrame(data_for_plot)
            
            # Vẽ biểu đồ
            plt.figure(figsize=(14, 10))
            
            # Sử dụng seaborn để vẽ biểu đồ nhiệt heat map
            pivot_table = plot_df.pivot_table(values='Score', index='Question Type', columns='Criteria')
            
            # Sắp xếp Question Type theo điểm trung bình để dễ so sánh
            if 'Average' in pivot_table.columns:
                avg_scores = pivot_table['Average']
                pivot_table = pivot_table.loc[avg_scores.sort_values(ascending=False).index]
            else:
                avg_scores = pivot_table.mean(axis=1)
                pivot_table = pivot_table.loc[avg_scores.sort_values(ascending=False).index]
            
            # Vẽ heatmap với annotation
            sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5,
                      vmin=1, vmax=5, cbar_kws={'label': 'Score (1-5 scale)'})
            
            plt.title('Reasoning Quality by Question Type', fontsize=18, pad=20)
            plt.tight_layout()
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"reasoning_by_question_type_{self.timestamp}.png")
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ reasoning theo loại câu hỏi: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("Reasoning by Question Type", f"Error: {str(e)}")
    
    def _create_reasoning_by_question_type_by_model_plot(self):
        """
        Tạo biểu đồ đánh giá reasoning theo loại câu hỏi và mô hình.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        # Kiểm tra điều kiện cần thiết
        if 'question_type' not in self.results_df.columns or 'model_name' not in self.results_df.columns:
            return self._create_fallback_plot("Reasoning by Question Type and Model", 
                                             "Missing question_type or model_name data")
            
        # Kiểm tra xem có dữ liệu reasoning không
        reasoning_cols = [col for col in self.results_df.columns if col.startswith('reasoning_') 
                         and col not in ['reasoning_evaluation', 'reasoning_scores_str']]
        
        if not reasoning_cols:
            return self._create_fallback_plot("Reasoning by Question Type and Model", 
                                             "No reasoning data available")
        
        try:
            # Lấy danh sách các model và loại câu hỏi
            models = self.results_df['model_name'].unique()
            question_types = self.results_df['question_type'].unique()
            
            # Tập trung vào điểm trung bình để đơn giản hoá biểu đồ
            if 'reasoning_average' not in reasoning_cols:
                # Nếu không có cột reasoning_average, sử dụng cột reasoning đầu tiên
                target_col = reasoning_cols[0]
                col_label = target_col.replace('reasoning_', '').title()
            else:
                target_col = 'reasoning_average'
                col_label = 'Average Reasoning'
            
            # Tạo dữ liệu cho biểu đồ
            data_for_plot = []
            for model in models:
                for q_type in question_types:
                    # Lọc dữ liệu theo model và loại câu hỏi
                    filtered_df = self.results_df[(self.results_df['model_name'] == model) & 
                                                (self.results_df['question_type'] == q_type)]
                    
                    if not filtered_df.empty and target_col in filtered_df.columns:
                        # Tính điểm trung bình
                        avg_score = filtered_df[target_col].mean()
                        sample_count = len(filtered_df)
                        
                        if not pd.isna(avg_score) and sample_count > 0:
                            data_for_plot.append({
                                'Model': model,
                                'Question Type': q_type,
                                'Score': avg_score,
                                'Sample Count': sample_count
                            })
            
            if not data_for_plot:
                return self._create_fallback_plot("Reasoning by Question Type and Model", 
                                                "Insufficient reasoning data")
            
            # Tạo DataFrame cho biểu đồ
            plot_df = pd.DataFrame(data_for_plot)
            
            # Vẽ heat map để thể hiện mối quan hệ giữa model và loại câu hỏi
            plt.figure(figsize=(16, 12))
            
            # Tạo pivot table
            pivot_table = plot_df.pivot_table(values='Score', 
                                            index='Model', 
                                            columns='Question Type')
            
            # Vẽ heatmap với annotation
            sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5,
                      vmin=1, vmax=5, cbar_kws={'label': 'Score (1-5 scale)'})
            
            plt.title(f'Reasoning Quality by Question Type and Model ({col_label})', 
                     fontsize=18, pad=20)
            plt.tight_layout()
            
            # Lưu biểu đồ chính
            output_path = os.path.join(self.plots_dir, 
                                     f"reasoning_by_question_type_by_model_{self.timestamp}.png")
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            # Tạo biểu đồ bar để so sánh các model theo từng loại câu hỏi
            plt.figure(figsize=(18, len(question_types) * 3))
            
            # Tạo subplots cho mỗi loại câu hỏi
            fig, axes = plt.subplots(len(question_types), 1, 
                                   figsize=(14, len(question_types) * 3), 
                                   sharex=True)
            
            if len(question_types) == 1:
                axes = [axes]  # Đảm bảo axes luôn là một list
                
            for i, q_type in enumerate(question_types):
                # Lọc dữ liệu cho loại câu hỏi này
                type_data = plot_df[plot_df['Question Type'] == q_type]
                
                if not type_data.empty:
                    # Sắp xếp theo điểm số
                    type_data = type_data.sort_values('Score', ascending=False)
                    
                    # Vẽ biểu đồ bar - FIX: thêm hue='Model' và legend=False thay vì chỉ dùng palette
                    sns.barplot(x='Score', y='Model', hue='Model', data=type_data, ax=axes[i], 
                             palette='viridis', legend=False)
                    
                    # Thêm giá trị vào thanh
                    for j, v in enumerate(type_data['Score']):
                        sample_count = type_data.iloc[j]['Sample Count']
                        axes[i].text(v + 0.1, j, f"{v:.2f} (n={sample_count})", 
                                   va='center', fontsize=10)
                    
                    # Thêm tiêu đề và định dạng
                    axes[i].set_title(f'Question Type: {q_type}', fontsize=14)
                    axes[i].set_xlim(0, 5.5)  # Đặt thang điểm 0-5
                    axes[i].grid(axis='x', linestyle='--', alpha=0.7)
            
            # Thêm tiêu đề chung và định dạng
            fig.suptitle(f'Model Reasoning Performance by Question Type ({col_label})', 
                        fontsize=18)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Lưu biểu đồ bar
            bar_output_path = os.path.join(self.plots_dir, 
                                         f"reasoning_by_question_type_by_model_bar_{self.timestamp}.png")
            plt.savefig(bar_output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            # Nếu có nhiều cột reasoning, tạo thêm biểu đồ radar cho từng model
            if len(reasoning_cols) >= 3 and 'reasoning_average' in reasoning_cols:
                # Tạo dữ liệu cho radar chart
                criteria_labels = {
                    'reasoning_accuracy': 'Accuracy',
                    'reasoning_logic': 'Logic',
                    'reasoning_consistency': 'Consistency',
                    'reasoning_difficulty': 'Difficulty',
                    'reasoning_context': 'Context'
                }
                
                # Lọc các cột tiêu chí chính, bỏ qua average
                radar_cols = [col for col in reasoning_cols if col in criteria_labels]
                
                if len(radar_cols) >= 3:  # Cần ít nhất 3 tiêu chí để vẽ radar chart
                    # Tạo một biểu đồ radar cho mỗi model
                    for model in models:
                        model_data = {}
                        
                        # Tính điểm trung bình cho mỗi tiêu chí theo model
                        model_df = self.results_df[self.results_df['model_name'] == model]
                        
                        for col in radar_cols:
                            if col in model_df.columns:
                                label = criteria_labels.get(col, col.replace('reasoning_', '').title())
                                avg_score = model_df[col].mean()
                                if not pd.isna(avg_score):
                                    model_data[label] = avg_score
                        
                        if len(model_data) >= 3:  # Đảm bảo có đủ dữ liệu
                            # Chuẩn bị dữ liệu cho radar chart
                            categories = list(model_data.keys())
                            values = list(model_data.values())
                            
                            # Tạo biểu đồ radar
                            plt.figure(figsize=(10, 8))
                            
                            # Tính toán góc cho mỗi trục
                            N = len(categories)
                            angles = [n / float(N) * 2 * np.pi for n in range(N)]
                            angles += angles[:1]  # Đóng hình đa giác
                            
                            # Thêm giá trị
                            values += values[:1]  # Đóng đa giác
                            
                            # Thiết lập biểu đồ
                            ax = plt.subplot(111, polar=True)
                            
                            # Vẽ đường biểu đồ
                            ax.plot(angles, values, linewidth=2, linestyle='solid')
                            ax.fill(angles, values, alpha=0.4)
                            
                            # Thiết lập các trục và nhãn
                            plt.xticks(angles[:-1], categories, fontsize=12)
                            
                            # Thiết lập giới hạn y
                            ax.set_ylim(0, 5)
                            
                            # Thêm nhãn giá trị
                            ax.set_rlabel_position(0)
                            plt.yticks([1, 2, 3, 4, 5], ['1', '2', '3', '4', '5'], fontsize=10)
                            
                            # Thêm tiêu đề
                            plt.title(f"Reasoning Criteria Performance: {model}", fontsize=15)
                            
                            # Lưu biểu đồ
                            radar_path = os.path.join(self.plots_dir, 
                                                    f"reasoning_radar_{model}_{self.timestamp}.png")
                            plt.savefig(radar_path, dpi=120, bbox_inches='tight')
                            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ reasoning theo loại câu hỏi và model: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("Reasoning by Question Type and Model", f"Error: {str(e)}")
    
    def _create_f1_score_plot(self):
        """
        Tạo biểu đồ F1 Score theo model và prompt type.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'f1_score' not in self.results_df.columns:
            return self._create_fallback_plot("F1 Score", "No F1 score data available")
        
        try:
            # Kiểm tra số lượng giá trị không phải NaN
            non_nan_count = self.results_df['f1_score'].notna().sum()
            if non_nan_count == 0:
                logger.warning("Không có giá trị F1 score hợp lệ nào để vẽ biểu đồ")
                return self._create_fallback_plot("F1 Score", "No valid F1 score values available")
                
            logger.info(f"Tìm thấy {non_nan_count} giá trị F1 score hợp lệ để vẽ biểu đồ")
            
            # Tính F1 score trung bình theo model và prompt
            f1_by_model_prompt = self.results_df.groupby(['model_name', 'prompt_type'])['f1_score'].mean().unstack()
            
            # Kiểm tra xem có đủ dữ liệu để vẽ biểu đồ không
            if f1_by_model_prompt.empty or f1_by_model_prompt.isna().all().all():
                logger.warning("Không đủ dữ liệu F1 score để vẽ biểu đồ")
                return self._create_fallback_plot("F1 Score", "Insufficient data for F1 score visualization")
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Xác định giới hạn giá trị dựa trên dữ liệu thực tế
            vmin = np.floor(f1_by_model_prompt.min().min() * 10) / 10 if not np.isnan(f1_by_model_prompt.min().min()) else 0
            vmax = np.ceil(f1_by_model_prompt.max().max() * 10) / 10 if not np.isnan(f1_by_model_prompt.max().max()) else 1
            
            # Đảm bảo vmin và vmax nằm trong khoảng [0, 1]
            vmin = max(0, min(vmin, 0.9))
            vmax = min(1.0, max(vmax, 0.1))
            
            # Log để debug
            logger.debug(f"F1 score range: vmin={vmin}, vmax={vmax}")
            
            # Sử dụng seaborn heatmap với định dạng số tùy chỉnh
            sns.heatmap(f1_by_model_prompt, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5,
                      vmin=vmin, vmax=vmax, cbar_kws={'label': 'F1 Score'}, annot_kws={"size": 10})
            
            plt.title('F1 Score by Model and Prompt Type', fontsize=18, pad=20)
            plt.tight_layout()
            
            # Thêm chú thích về ý nghĩa của F1 Score
            plt.figtext(0.5, 0.01, 
                       "F1 Score đo lường độ tương đồng về mặt từ vựng (token overlap) giữa câu trả lời và đáp án chuẩn.\nGiá trị cao hơn thể hiện sự tương đồng tốt hơn.", 
                       ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.15, "pad":5})
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"f1_score_{self.timestamp}.png")
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ F1 score: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("F1 Score", f"Error: {str(e)}")
    
    def _create_meteor_score_plot(self):
        """
        Tạo biểu đồ METEOR Score theo model và prompt type.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'meteor_score' not in self.results_df.columns:
            return self._create_fallback_plot("METEOR Score", "No METEOR score data available")
        
        try:
            # Kiểm tra số lượng giá trị không phải NaN
            non_nan_count = self.results_df['meteor_score'].notna().sum()
            if non_nan_count == 0:
                logger.warning("Không có giá trị METEOR score hợp lệ nào để vẽ biểu đồ")
                return self._create_fallback_plot("METEOR Score", "No valid METEOR score values available")
                
            logger.info(f"Tìm thấy {non_nan_count} giá trị METEOR score hợp lệ để vẽ biểu đồ")
            
            # Tính METEOR score trung bình theo model và prompt
            meteor_by_model_prompt = self.results_df.groupby(['model_name', 'prompt_type'])['meteor_score'].mean().unstack()
            
            # Kiểm tra xem có đủ dữ liệu để vẽ biểu đồ không
            if meteor_by_model_prompt.empty or meteor_by_model_prompt.isna().all().all():
                logger.warning("Không đủ dữ liệu METEOR score để vẽ biểu đồ")
                return self._create_fallback_plot("METEOR Score", "Insufficient data for METEOR score visualization")
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Xác định giới hạn giá trị dựa trên dữ liệu thực tế
            vmin = np.floor(meteor_by_model_prompt.min().min() * 10) / 10 if not np.isnan(meteor_by_model_prompt.min().min()) else 0
            vmax = np.ceil(meteor_by_model_prompt.max().max() * 10) / 10 if not np.isnan(meteor_by_model_prompt.max().max()) else 1
            
            # Đảm bảo vmin và vmax nằm trong khoảng [0, 1]
            vmin = max(0, min(vmin, 0.9))
            vmax = min(1.0, max(vmax, 0.1))
            
            # Log để debug
            logger.debug(f"METEOR score range: vmin={vmin}, vmax={vmax}")
            
            # Sử dụng seaborn heatmap với định dạng số tùy chỉnh
            sns.heatmap(meteor_by_model_prompt, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5,
                      vmin=vmin, vmax=vmax, cbar_kws={'label': 'METEOR Score'}, annot_kws={"size": 10})
            
            plt.title('METEOR Score by Model and Prompt Type', fontsize=18, pad=20)
            plt.tight_layout()
            
            # Thêm chú thích về ý nghĩa của METEOR Score
            plt.figtext(0.5, 0.01, 
                       "METEOR Score đo lường chất lượng tương đồng thông qua các tiêu chí đồng nghĩa, từ vựng và cấu trúc.\nĐược dùng nhiều trong đánh giá dịch thuật, giá trị cao hơn thể hiện sự tương đồng tốt hơn.", 
                       ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.15, "pad":5})
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"meteor_score_{self.timestamp}.png")
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ METEOR score: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("METEOR Score", f"Error: {str(e)}")
    
    def _create_bertscore_plot(self):
        """
        Tạo biểu đồ BERT Score theo model và prompt type.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'bert_score' not in self.results_df.columns:
            return self._create_fallback_plot("BERT Score", "No BERT score data available")
        
        try:
            # Kiểm tra số lượng giá trị không phải NaN
            non_nan_count = self.results_df['bert_score'].notna().sum()
            if non_nan_count == 0:
                logger.warning("Không có giá trị BERT score hợp lệ nào để vẽ biểu đồ")
                return self._create_fallback_plot("BERT Score", "No valid BERT score values available")
                
            logger.info(f"Tìm thấy {non_nan_count} giá trị BERT score hợp lệ để vẽ biểu đồ")
            
            # Tính BERT score trung bình theo model và prompt
            bert_by_model_prompt = self.results_df.groupby(['model_name', 'prompt_type'])['bert_score'].mean().unstack()
            
            # Kiểm tra xem có đủ dữ liệu để vẽ biểu đồ không
            if bert_by_model_prompt.empty or bert_by_model_prompt.isna().all().all():
                logger.warning("Không đủ dữ liệu BERT score để vẽ biểu đồ")
                return self._create_fallback_plot("BERT Score", "Insufficient data for BERT score visualization")
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Xác định giới hạn giá trị dựa trên dữ liệu thực tế
            vmin = np.floor(bert_by_model_prompt.min().min() * 10) / 10 if not np.isnan(bert_by_model_prompt.min().min()) else 0
            vmax = np.ceil(bert_by_model_prompt.max().max() * 10) / 10 if not np.isnan(bert_by_model_prompt.max().max()) else 1
            
            # Đảm bảo vmin và vmax nằm trong khoảng [0, 1]
            vmin = max(0, min(vmin, 0.9))
            vmax = min(1.0, max(vmax, 0.1))
            
            # Log để debug
            logger.debug(f"BERT score range: vmin={vmin}, vmax={vmax}")
            
            # Định dạng số hiển thị trong heatmap để dễ đọc
            def fmt(x):
                if np.isnan(x):
                    return "N/A"
                else:
                    return f'{x:.3f}'
            
            # Sử dụng seaborn heatmap với định dạng số tùy chỉnh
            sns.heatmap(bert_by_model_prompt, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5,
                      vmin=vmin, vmax=vmax, cbar_kws={'label': 'BERT Score'}, annot_kws={"size": 10})
            
            plt.title('BERT Score by Model and Prompt Type', fontsize=18, pad=20)
            plt.tight_layout()
            
            # Thêm chú thích về ý nghĩa của BERT Score
            plt.figtext(0.5, 0.01, 
                       "BERT Score đo lường độ tương đồng về mặt ngữ nghĩa giữa câu trả lời và đáp án chuẩn.\nGiá trị cao hơn thể hiện sự tương đồng tốt hơn.", 
                       ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.15, "pad":5})
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"bert_score_{self.timestamp}.png")
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ BERT score: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("BERT Score", f"Error: {str(e)}")

    def _create_exact_match_plot(self):
        """
        Tạo biểu đồ Exact Match (EM) Score theo model và prompt type.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'exact_match' not in self.results_df.columns and 'em_score' not in self.results_df.columns:
            return self._create_fallback_plot("Exact Match Score", "No Exact Match score data available")
        
        try:
            # Xác định tên cột EM
            em_column = 'exact_match' if 'exact_match' in self.results_df.columns else 'em_score'
            
            # Kiểm tra số lượng giá trị không phải NaN
            non_nan_count = self.results_df[em_column].notna().sum()
            if non_nan_count == 0:
                logger.warning("Không có giá trị Exact Match score hợp lệ nào để vẽ biểu đồ")
                return self._create_fallback_plot("Exact Match Score", "No valid Exact Match score values available")
                
            logger.info(f"Tìm thấy {non_nan_count} giá trị Exact Match score hợp lệ để vẽ biểu đồ")
            
            # Tính EM score trung bình theo model và prompt
            em_by_model_prompt = self.results_df.groupby(['model_name', 'prompt_type'])[em_column].mean().unstack()
            
            # Kiểm tra xem có đủ dữ liệu để vẽ biểu đồ không
            if em_by_model_prompt.empty or em_by_model_prompt.isna().all().all():
                logger.warning("Không đủ dữ liệu Exact Match score để vẽ biểu đồ")
                return self._create_fallback_plot("Exact Match Score", "Insufficient data for Exact Match score visualization")
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Xác định giới hạn giá trị dựa trên dữ liệu thực tế
            vmin = 0  # Exact Match thường nằm trong khoảng [0, 1]
            vmax = 1.0
            
            # Sử dụng seaborn heatmap với định dạng số tùy chỉnh
            sns.heatmap(em_by_model_prompt, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5,
                      vmin=vmin, vmax=vmax, cbar_kws={'label': 'Exact Match Score'}, annot_kws={"size": 10})
            
            plt.title('Exact Match Score by Model and Prompt Type', fontsize=18, pad=20)
            plt.tight_layout()
            
            # Thêm chú thích về ý nghĩa của Exact Match Score
            plt.figtext(0.5, 0.01, 
                       "Exact Match đánh giá sự khớp chính xác giữa câu trả lời và đáp án chuẩn.\nGiá trị 1.0 = khớp hoàn toàn, 0.0 = không khớp.", 
                       ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.15, "pad":5})
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"exact_match_score_{self.timestamp}.png")
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ Exact Match score: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("Exact Match Score", f"Error: {str(e)}")

    def _create_rouge_scores_plot(self):
        """
        Tạo biểu đồ ROUGE Scores theo model và prompt type.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        # Kiểm tra các cột ROUGE khác nhau có thể có
        rouge_columns = [col for col in self.results_df.columns if 'rouge' in col.lower()]
        
        if not rouge_columns:
            return self._create_fallback_plot("ROUGE Scores", "No ROUGE scores data available")
        
        try:
            # Ưu tiên ROUGE-L nếu có, hoặc ROUGE-1
            if 'rougeL_f' in rouge_columns or 'rouge_l' in rouge_columns:
                rouge_col = 'rougeL_f' if 'rougeL_f' in rouge_columns else 'rouge_l'
            elif 'rouge1_f' in rouge_columns or 'rouge_1' in rouge_columns:
                rouge_col = 'rouge1_f' if 'rouge1_f' in rouge_columns else 'rouge_1'
            else:
                # Sử dụng cột ROUGE đầu tiên tìm thấy
                rouge_col = rouge_columns[0]
            
            # Kiểm tra số lượng giá trị không phải NaN
            non_nan_count = self.results_df[rouge_col].notna().sum()
            if non_nan_count == 0:
                logger.warning(f"Không có giá trị {rouge_col} hợp lệ nào để vẽ biểu đồ")
                return self._create_fallback_plot("ROUGE Scores", f"No valid {rouge_col} values available")
                
            logger.info(f"Tìm thấy {non_nan_count} giá trị {rouge_col} hợp lệ để vẽ biểu đồ")
            
            # Tính ROUGE score trung bình theo model và prompt
            rouge_by_model_prompt = self.results_df.groupby(['model_name', 'prompt_type'])[rouge_col].mean().unstack()
            
            # Kiểm tra xem có đủ dữ liệu để vẽ biểu đồ không
            if rouge_by_model_prompt.empty or rouge_by_model_prompt.isna().all().all():
                logger.warning(f"Không đủ dữ liệu {rouge_col} để vẽ biểu đồ")
                return self._create_fallback_plot("ROUGE Scores", f"Insufficient data for {rouge_col} visualization")
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Xác định giới hạn giá trị dựa trên dữ liệu thực tế
            vmin = np.floor(rouge_by_model_prompt.min().min() * 10) / 10 if not np.isnan(rouge_by_model_prompt.min().min()) else 0
            vmax = np.ceil(rouge_by_model_prompt.max().max() * 10) / 10 if not np.isnan(rouge_by_model_prompt.max().max()) else 1
            
            # Đảm bảo vmin và vmax nằm trong khoảng [0, 1]
            vmin = max(0, min(vmin, 0.9))
            vmax = min(1.0, max(vmax, 0.1))
            
            # Sử dụng seaborn heatmap
            sns.heatmap(rouge_by_model_prompt, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5,
                      vmin=vmin, vmax=vmax, cbar_kws={'label': f'{rouge_col.upper()}'}, annot_kws={"size": 10})
            
            # Thiết lập tiêu đề dựa vào loại ROUGE được sử dụng
            rouge_type = rouge_col.split('_')[0].upper() if '_' in rouge_col else rouge_col.upper()
            plt.title(f'{rouge_type} Scores by Model and Prompt Type', fontsize=18, pad=20)
            plt.tight_layout()
            
            # Thêm chú thích về ý nghĩa của ROUGE Score
            if 'rougeL' in rouge_col or 'rouge_l' in rouge_col:
                description = "ROUGE-L đánh giá chuỗi con chung dài nhất, tập trung vào cấu trúc câu và thứ tự từ.\nGiá trị cao hơn thể hiện sự tương đồng tốt hơn."
            elif 'rouge1' in rouge_col or 'rouge_1' in rouge_col:
                description = "ROUGE-1 đánh giá sự trùng lặp unigram (từ đơn) giữa câu trả lời và đáp án chuẩn.\nGiá trị cao hơn thể hiện sự tương đồng tốt hơn."
            elif 'rouge2' in rouge_col or 'rouge_2' in rouge_col:
                description = "ROUGE-2 đánh giá sự trùng lặp bigram (cặp từ) giữa câu trả lời và đáp án chuẩn.\nGiá trị cao hơn thể hiện sự tương đồng tốt hơn."
            else:
                description = "ROUGE đánh giá sự tương đồng giữa câu trả lời và đáp án chuẩn.\nGiá trị cao hơn thể hiện sự tương đồng tốt hơn."
            
            plt.figtext(0.5, 0.01, description, ha="center", fontsize=10, 
                      bbox={"facecolor":"orange", "alpha":0.15, "pad":5})
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"rouge_scores_{self.timestamp}.png")
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ ROUGE score: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("ROUGE Scores", f"Error: {str(e)}")
    
    def _create_bleu_scores_plot(self):
        """
        Tạo biểu đồ BLEU Scores theo model và prompt type.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        # Kiểm tra các cột BLEU khác nhau có thể có
        bleu_columns = [col for col in self.results_df.columns if 'bleu' in col.lower()]
        
        if not bleu_columns:
            return self._create_fallback_plot("BLEU Scores", "No BLEU scores data available")
        
        try:
            # Ưu tiên BLEU tổng hợp, sau đó là BLEU-1, BLEU-2, vv
            if 'bleu' in bleu_columns:
                bleu_col = 'bleu'  # BLEU tổng hợp
            elif 'bleu1' in bleu_columns:
                bleu_col = 'bleu1'  # BLEU-1
            else:
                # Sử dụng cột BLEU đầu tiên tìm thấy
                bleu_col = bleu_columns[0]
            
            # Kiểm tra số lượng giá trị không phải NaN
            non_nan_count = self.results_df[bleu_col].notna().sum()
            if non_nan_count == 0:
                logger.warning(f"Không có giá trị {bleu_col} hợp lệ nào để vẽ biểu đồ")
                return self._create_fallback_plot("BLEU Scores", f"No valid {bleu_col} values available")
                
            logger.info(f"Tìm thấy {non_nan_count} giá trị {bleu_col} hợp lệ để vẽ biểu đồ")
            
            # Tính BLEU score trung bình theo model và prompt
            bleu_by_model_prompt = self.results_df.groupby(['model_name', 'prompt_type'])[bleu_col].mean().unstack()
            
            # Kiểm tra xem có đủ dữ liệu để vẽ biểu đồ không
            if bleu_by_model_prompt.empty or bleu_by_model_prompt.isna().all().all():
                logger.warning(f"Không đủ dữ liệu {bleu_col} để vẽ biểu đồ")
                return self._create_fallback_plot("BLEU Scores", f"Insufficient data for {bleu_col} visualization")
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Xác định giới hạn giá trị dựa trên dữ liệu thực tế
            vmin = np.floor(bleu_by_model_prompt.min().min() * 10) / 10 if not np.isnan(bleu_by_model_prompt.min().min()) else 0
            vmax = np.ceil(bleu_by_model_prompt.max().max() * 10) / 10 if not np.isnan(bleu_by_model_prompt.max().max()) else 1
            
            # Đảm bảo vmin và vmax nằm trong khoảng [0, 1]
            vmin = max(0, min(vmin, 0.9))
            vmax = min(1.0, max(vmax, 0.1))
            
            # Sử dụng seaborn heatmap
            sns.heatmap(bleu_by_model_prompt, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5,
                      vmin=vmin, vmax=vmax, cbar_kws={'label': f'{bleu_col.upper()} Score'}, annot_kws={"size": 10})
            
            # Thiết lập tiêu đề dựa vào loại BLEU được sử dụng
            bleu_type = bleu_col.upper()
            plt.title(f'{bleu_type} Scores by Model and Prompt Type', fontsize=18, pad=20)
            plt.tight_layout()
            
            # Thêm chú thích về ý nghĩa của BLEU Score
            if bleu_col == 'bleu':
                description = "BLEU đánh giá mức độ tương đồng n-gram giữa câu trả lời và đáp án chuẩn.\nThường dùng trong đánh giá dịch máy, giá trị cao hơn thể hiện sự tương đồng tốt hơn."
            elif bleu_col == 'bleu1':
                description = "BLEU-1 đánh giá sự trùng lặp unigram (từ đơn) giữa câu trả lời và đáp án chuẩn.\nGiá trị cao hơn thể hiện sự tương đồng tốt hơn."
            elif bleu_col == 'bleu2':
                description = "BLEU-2 đánh giá sự trùng lặp bigram (cặp từ) giữa câu trả lời và đáp án chuẩn.\nGiá trị cao hơn thể hiện sự tương đồng tốt hơn."
            elif bleu_col == 'bleu3':
                description = "BLEU-3 đánh giá sự trùng lặp trigram (bộ ba từ) giữa câu trả lời và đáp án chuẩn.\nGiá trị cao hơn thể hiện sự tương đồng tốt hơn."
            elif bleu_col == 'bleu4':
                description = "BLEU-4 đánh giá sự trùng lặp 4-gram (bộ bốn từ) giữa câu trả lời và đáp án chuẩn.\nGiá trị cao hơn thể hiện sự tương đồng tốt hơn."
            else:
                description = f"{bleu_col.upper()} đánh giá mức độ tương đồng giữa câu trả lời và đáp án chuẩn.\nGiá trị cao hơn thể hiện sự tương đồng tốt hơn."
            
            plt.figtext(0.5, 0.01, description, ha="center", fontsize=10, 
                      bbox={"facecolor":"orange", "alpha":0.15, "pad":5})
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"bleu_scores_{self.timestamp}.png")
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ BLEU score: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("BLEU Scores", f"Error: {str(e)}")

    def _generate_visualizations(self) -> Dict[str, str]:
        """
        Tạo tất cả các biểu đồ trực quan hóa cho báo cáo.
        
        Returns:
            Dict[str, str]: Dictionary chứa đường dẫn đến các biểu đồ
        """
        plot_paths = {}
        
        def create_plot(plot_function, plot_name, description):
            logger.info(f"Đang tạo biểu đồ: {plot_name}")
            try:
                path = plot_function()
                if path:
                    plot_paths[plot_name] = {
                        'path': path,
                        'description': description
                    }
                    logger.info(f"Đã tạo biểu đồ {plot_name}: {path}")
                else:
                    logger.warning(f"Không thể tạo biểu đồ {plot_name}")
            except Exception as e:
                logger.error(f"Lỗi khi tạo biểu đồ {plot_name}: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Các biểu đồ cơ bản (giữ nguyên)
        create_plot(self._create_accuracy_by_model_plot, 'accuracy_by_model', 
                   'Accuracy trung bình theo từng model')
        
        create_plot(self._create_accuracy_by_prompt_plot, 'accuracy_by_prompt',
                   'Accuracy trung bình theo từng loại prompt')
        
        create_plot(self._create_accuracy_heatmap, 'accuracy_heatmap',
                   'Accuracy chi tiết theo model và prompt')
        
        create_plot(self._create_simple_comparison_plot, 'simple_comparison',
                   'So sánh hiệu suất tổng thể giữa các model')
        
        # Thêm các biểu đồ đánh giá reasoning
        create_plot(self._create_reasoning_criteria_plot, 'reasoning_criteria',
                   'Đánh giá các tiêu chí suy luận theo model')
        
        create_plot(self._create_reasoning_by_prompt_plot, 'reasoning_by_prompt',
                   'Chất lượng suy luận trung bình theo loại prompt')
        
        # Thêm biểu đồ reasoning theo loại câu hỏi
        create_plot(self._create_reasoning_by_question_type_plot, 'reasoning_by_question_type',
                   'Chất lượng suy luận phân theo loại câu hỏi')
                   
        # Thêm biểu đồ reasoning theo loại câu hỏi và model
        create_plot(self._create_reasoning_by_question_type_by_model_plot, 'reasoning_by_question_type_by_model',
                   'Chất lượng suy luận phân theo loại câu hỏi và model')
                   
        # Thêm biểu đồ consistency score
        create_plot(self._create_consistency_score_plot, 'consistency_score',
                   'Đánh giá tính nhất quán (consistency) trong các câu trả lời của model')
        
        # Thêm biểu đồ phân tích lỗi (error analysis)
        create_plot(self._create_error_analysis_plot, 'error_analysis',
                  'Phân tích và phân loại các lỗi trong câu trả lời của model')
        
        # Thêm các biểu đồ đánh giá theo criteria khác
        create_plot(self._create_criteria_evaluation_plot, 'criteria_evaluation',
                   'Đánh giá theo các tiêu chí chất lượng')
        
        create_plot(self._create_criteria_radar_plot, 'criteria_radar',
                   'Đánh giá đa tiêu chí theo dạng radar chart')
        
        create_plot(self._create_difficulty_performance_plot, 'difficulty_performance',
                   'Hiệu suất trên các câu hỏi có độ khó khác nhau')
        
        create_plot(self._create_context_adherence_plot, 'context_adherence',
                   'Độ phù hợp ngữ cảnh theo model và prompt')
        
        # Thêm các biểu đồ metrics nâng cao
        create_plot(self._create_exact_match_plot, 'exact_match',
                   'Exact Match Score đánh giá sự khớp chính xác giữa câu trả lời và đáp án')
                   
        create_plot(self._create_rouge_scores_plot, 'rouge_scores',
                   'ROUGE Score đánh giá độ tương đồng văn bản và chất lượng tóm tắt')
                   
        create_plot(self._create_bleu_scores_plot, 'bleu_scores',
                   'BLEU Score đánh giá chất lượng dịch thuật và sinh văn bản')
                   
        create_plot(self._create_f1_score_plot, 'f1_score',
                   'F1 Score dựa trên sự trùng lặp token')
        
        create_plot(self._create_meteor_score_plot, 'meteor_score',
                   'METEOR Score đánh giá chất lượng dịch thuật')
        
        create_plot(self._create_bertscore_plot, 'bert_score',
                   'BERT Score đánh giá độ tương đồng ngữ nghĩa')
                   
        return plot_paths

    def _create_consistency_score_plot(self):
        """
        Tạo biểu đồ đánh giá tính nhất quán của các mô hình.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        # Kiểm tra xem có dữ liệu consistency không
        if 'consistency_score' not in self.results_df.columns:
            return self._create_fallback_plot("Consistency Score", "No consistency score data available")
        
        # Lọc các dòng có giá trị consistency_score không phải NaN
        consistency_df = self.results_df[~self.results_df['consistency_score'].isna()]
        
        if len(consistency_df) == 0:
            return self._create_fallback_plot("Consistency Score", "No valid consistency score values available")
        
        try:
            # 1. Tạo biểu đồ tổng quan về consistency score
            plt.figure(figsize=(14, 10))
            
            # Sử dụng model_name thay vì model nếu có
            model_col = 'model_name' if 'model_name' in consistency_df.columns else 'model'
            
            # Tính consistency score trung bình theo model và prompt type
            if model_col in consistency_df.columns and 'prompt_type' in consistency_df.columns:
                pivot_df = consistency_df.pivot_table(
                    values='consistency_score',
                    index=model_col,
                    columns='prompt_type',
                    aggfunc='mean'
                )
                
                # Vẽ heatmap
                ax = sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f", 
                               linewidths=.5, vmin=0, vmax=1.0, cbar_kws={'label': 'Consistency Score'})
                plt.title('Model Consistency Score by Prompt Type', fontsize=16)
                plt.ylabel('Model')
                plt.xlabel('Prompt Type')
                
                # Thêm chú thích
                plt.figtext(0.5, 0.01, 
                           "Consistency Score đo lường mức độ nhất quán trong câu trả lời của model khi chạy nhiều lần.\n1.0 = hoàn toàn nhất quán, 0.0 = hoàn toàn không nhất quán.",
                           ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
                
                # Lưu biểu đồ
                consistency_score_path = os.path.join(self.plots_dir, f"consistency_score_{self.timestamp}.png")
                plt.savefig(consistency_score_path, dpi=120, bbox_inches='tight')
                plt.close()
                
                # 2. Tạo biểu đồ thứ hai về agreement rate
                plt.figure(figsize=(14, 10))
                
                # Tính agreement rate trung bình theo model và prompt type
                pivot_df_agreement = consistency_df.pivot_table(
                    values='consistency_agreement_rate',
                    index=model_col,
                    columns='prompt_type',
                    aggfunc='mean'
                )
                
                # Vẽ heatmap
                sns.heatmap(pivot_df_agreement, annot=True, cmap="YlGnBu", fmt=".3f", 
                          linewidths=.5, vmin=0, vmax=1.0, cbar_kws={'label': 'Agreement Rate'})
                plt.title('Model Agreement Rate by Prompt Type', fontsize=16)
                plt.ylabel('Model')
                plt.xlabel('Prompt Type')
                
                # Thêm chú thích
                plt.figtext(0.5, 0.01, 
                           "Agreement Rate là tỷ lệ các lần chạy cho ra câu trả lời phổ biến nhất.\nChỉ số này đo lường khả năng đưa ra cùng một câu trả lời của model.",
                           ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
                
                # Lưu biểu đồ
                agreement_rate_path = os.path.join(self.plots_dir, f"agreement_rate_{self.timestamp}.png")
                plt.savefig(agreement_rate_path, dpi=120, bbox_inches='tight')
                plt.close()
                
                # 3. Tạo biểu đồ so sánh tất cả các model
                plt.figure(figsize=(14, 10))
                
                # Tính giá trị trung bình theo model
                consistency_by_model = consistency_df.groupby(model_col)['consistency_score'].mean()
                agreement_by_model = consistency_df.groupby(model_col)['consistency_agreement_rate'].mean()
                
                # Tạo DataFrame mới để vẽ
                comparison_df = pd.DataFrame({
                    'Consistency Score': consistency_by_model,
                    'Agreement Rate': agreement_by_model
                })
                
                # Vẽ biểu đồ cột
                comparison_df.plot(kind='bar', figsize=(14, 8), width=0.8)
                plt.title('Consistency Metrics by Model', fontsize=16)
                plt.ylabel('Score (0-1)')
                plt.xlabel('Model')
                plt.ylim(0, 1.05)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend(loc='best')
                
                # Thêm giá trị lên đầu các cột
                for i, v in enumerate(comparison_df['Consistency Score']):
                    plt.text(i-0.2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
                for i, v in enumerate(comparison_df['Agreement Rate']):
                    plt.text(i+0.2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
                
                # Lưu biểu đồ
                comparison_path = os.path.join(self.plots_dir, f"consistency_comparison_{self.timestamp}.png")
                plt.savefig(comparison_path, dpi=120, bbox_inches='tight')
                plt.close()
                
                return consistency_score_path
                
            else:
                # Nếu không có cả model và prompt_type
                avg_consistency = consistency_df['consistency_score'].mean()
                avg_agreement = consistency_df['consistency_agreement_rate'].mean() if 'consistency_agreement_rate' in consistency_df.columns else 0
                
                # Tạo biểu đồ đơn giản
                plt.figure(figsize=(10, 6))
                metrics = ['Consistency Score', 'Agreement Rate']
                values = [avg_consistency, avg_agreement]
                
                plt.bar(metrics, values, color=['skyblue', 'lightgreen'])
                plt.title('Overall Consistency Metrics', fontsize=16)
                plt.ylim(0, 1.05)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Thêm giá trị lên đầu các cột
                for i, v in enumerate(values):
                    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
                
                # Lưu biểu đồ
                output_path = os.path.join(self.plots_dir, f"consistency_overall_{self.timestamp}.png")
                plt.savefig(output_path, dpi=120, bbox_inches='tight')
                plt.close()
                
                return output_path
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ consistency score: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("Consistency Score", f"Error: {str(e)}")

    def _create_error_analysis_plot(self):
        """
        Tạo biểu đồ phân tích lỗi dựa trên kết quả phân loại lỗi.
        Hiển thị tỉ lệ các loại lỗi khác nhau theo model và prompt type.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        # Kiểm tra xem có dữ liệu error_type không
        if 'error_type' not in self.results_df.columns:
            return self._create_fallback_plot("Error Analysis", "No error analysis data available")
        
        # Lọc các dòng có phân loại lỗi (error_type không rỗng và is_correct=False)
        error_df = self.results_df[(self.results_df['error_type'] != '') & 
                                  (self.results_df['is_correct'] == False)]
        
        if len(error_df) == 0:
            return self._create_fallback_plot("Error Analysis", "No error analysis data available")
        
        try:
            # Sử dụng model_name thay vì model nếu có
            model_col = 'model_name' if 'model_name' in error_df.columns else 'model'
            
            # 1. Tạo biểu đồ tổng quan về phân bố các loại lỗi
            plt.figure(figsize=(14, 10))
            
            # Đếm số lượng mỗi loại lỗi
            error_counts = error_df['error_type'].value_counts()
            
            # Tạo biểu đồ cột
            ax = error_counts.plot(kind='bar', color='lightcoral')
            plt.title('Phân bố các loại lỗi', fontsize=16)
            plt.xlabel('Loại lỗi')
            plt.ylabel('Số lượng')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Thêm giá trị lên đầu các cột
            for i, v in enumerate(error_counts):
                ax.text(i, v + 0.5, str(v), ha='center')
            
            # Thêm chú thích
            plt.figtext(0.5, 0.01, 
                       "Error Analysis phân loại các lỗi của model thành các nhóm như lỗi kiến thức, lỗi suy luận, lỗi tính toán, v.v.",
                       ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            
            # Lưu biểu đồ
            overall_path = os.path.join(self.plots_dir, f"error_analysis_overall_{self.timestamp}.png")
            plt.savefig(overall_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            # 2. Tạo biểu đồ phân tích lỗi theo model
            if model_col in error_df.columns:
                plt.figure(figsize=(14, 12))
                
                # Tạo crosstab để đếm số lượng mỗi loại lỗi theo model
                error_by_model = pd.crosstab(error_df[model_col], error_df['error_type'])
                
                # Tính phần trăm
                error_by_model_pct = error_by_model.div(error_by_model.sum(axis=1), axis=0) * 100
                
                # Vẽ heatmap
                sns.heatmap(error_by_model_pct, annot=error_by_model.values, fmt='d', cmap="YlOrRd", 
                          linewidths=0.5, cbar_kws={'label': 'Phần trăm (%)'})
                
                plt.title('Phân tích lỗi theo Model', fontsize=16)
                plt.ylabel('Model')
                plt.xlabel('Loại lỗi')
                plt.xticks(rotation=45, ha='right')
                
                # Lưu biểu đồ
                model_path = os.path.join(self.plots_dir, f"error_analysis_by_model_{self.timestamp}.png")
                plt.savefig(model_path, dpi=120, bbox_inches='tight')
                plt.close()
            
            # 3. Tạo biểu đồ phân tích lỗi theo prompt type
            plt.figure(figsize=(16, 12))
            
            # Tạo crosstab để đếm số lượng mỗi loại lỗi theo prompt type
            error_by_prompt = pd.crosstab(error_df['prompt_type'], error_df['error_type'])
            
            # Tính phần trăm
            error_by_prompt_pct = error_by_prompt.div(error_by_prompt.sum(axis=1), axis=0) * 100
            
            # Vẽ heatmap
            sns.heatmap(error_by_prompt_pct, annot=error_by_prompt.values, fmt='d', cmap="YlOrRd", 
                      linewidths=0.5, cbar_kws={'label': 'Phần trăm (%)'})
            
            plt.title('Phân tích lỗi theo Prompt Type', fontsize=16)
            plt.ylabel('Prompt Type')
            plt.xlabel('Loại lỗi')
            plt.xticks(rotation=45, ha='right')
            
            # Lưu biểu đồ
            prompt_path = os.path.join(self.plots_dir, f"error_analysis_by_prompt_{self.timestamp}.png")
            plt.savefig(prompt_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            # 4. Tạo biểu đồ tròn tổng quan
            plt.figure(figsize=(12, 12))
            
            # Tính phần trăm
            error_pct = error_counts / error_counts.sum() * 100
            
            # Vẽ biểu đồ tròn
            plt.pie(error_pct, labels=error_pct.index, autopct='%1.1f%%', startangle=90,
                  wedgeprops={'edgecolor': 'white', 'linewidth': 1},
                  textprops={'size': 12})
            
            plt.title('Tỷ lệ các loại lỗi', fontsize=16)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            # Thêm chú thích
            plt.figtext(0.5, 0.01, 
                       "Biểu đồ hiển thị tỷ lệ phần trăm của từng loại lỗi trong tổng số lỗi được phân tích",
                       ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            
            # Lưu biểu đồ
            pie_path = os.path.join(self.plots_dir, f"error_analysis_pie_{self.timestamp}.png")
            plt.savefig(pie_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            return overall_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ error analysis: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("Error Analysis", f"Error: {str(e)}")
