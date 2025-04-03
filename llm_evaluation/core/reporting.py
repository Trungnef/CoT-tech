"""
Reporting module cho hệ thống đánh giá LLM.
Tạo báo cáo dạng HTML, CSV và các biểu đồ visualizations từ kết quả đánh giá.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import re
from io import BytesIO
import base64
import time
import traceback

# Thiết lập cho matplotlib và seaborn
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Import các module cần thiết
try:
    import markdown
except ImportError:
    logging.warning("Thư viện 'markdown' không được cài đặt. Một số tính năng báo cáo có thể bị hạn chế.")

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
except ImportError:
    logging.warning("Thư viện 'jinja2' không được cài đặt. Một số tính năng báo cáo có thể bị hạn chế.")

# Thiết lập logging
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Tạo báo cáo từ kết quả đánh giá LLM.
    Bao gồm báo cáo HTML, CSV và các biểu đồ visualizations.
    """
    
    def __init__(self, 
                 results_df: pd.DataFrame,
                 output_dir: str = None,
                 timestamp: str = None,
                 report_title: str = "Báo Cáo Đánh Giá LLM",
                 language: str = "vietnamese",
                 visualization_enabled: bool = True,
                 theme: str = "light"):
        """
        Khởi tạo ReportGenerator.
        
        Args:
            results_df (pd.DataFrame): DataFrame chứa kết quả đánh giá
            output_dir (str): Thư mục đầu ra cho báo cáo (mặc định là thư mục run_timestamp)
            timestamp (str): Mốc thời gian cho tên file báo cáo
            report_title (str): Tiêu đề báo cáo
            language (str): Ngôn ngữ của báo cáo ("vietnamese" hoặc "english")
            visualization_enabled (bool): Có tạo biểu đồ không
            theme (str): Chủ đề giao diện báo cáo ("light" hoặc "dark")
        """
        self.results_df = results_df
        self.output_dir = output_dir or os.path.join("results", f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_title = report_title
        self.language = language.lower()
        self.visualization_enabled = visualization_enabled
        self.theme = theme
        
        # Sử dụng cấu trúc thư mục của run_dir
        self.reports_dir = os.path.join(self.output_dir, "reports")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.data_dir = os.path.join(self.output_dir, "analyzed_results")
        
        # Tạo các thư mục
        for dir_path in [self.reports_dir, self.plots_dir, self.data_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Tiền xử lý dữ liệu
        if 'model_name' in self.results_df.columns and 'model' not in self.results_df.columns:
            self.results_df['model'] = self.results_df['model_name']
        
        # Phân tích ban đầu
        self.accuracy_by_model_prompt = self._calculate_accuracy_by_model_prompt()
        self.metrics_by_model = self._calculate_metrics_by_model()
        
        # Các đường dẫn file đầu ra
        self.html_report_path = os.path.join(self.reports_dir, f"report_{self.timestamp}.html")
        self.csv_report_path = os.path.join(self.data_dir, f"report_summary_{self.timestamp}.csv")
        self.json_report_path = os.path.join(self.data_dir, f"report_summary_{self.timestamp}.json")
        
        # Template mặc định nếu không sử dụng file
        self._init_default_template()
    
    def generate_reports(self) -> Dict[str, str]:
        """
        Tạo tất cả các báo cáo: HTML, CSV và các biểu đồ visualizations.
        
        Returns:
            Dict[str, str]: Dictionary chứa đường dẫn đến các báo cáo được tạo
        """
        logger.info(f"Bắt đầu tạo báo cáo với {len(self.results_df)} kết quả")
        
        start_time = time.time()
        report_paths = {}
        
        # 1. Lưu dữ liệu phân tích dạng CSV và JSON
        csv_path = self._save_summary_csv()
        json_path = self._save_summary_json()
        report_paths['csv'] = csv_path
        report_paths['json'] = json_path
        
        # 2. Tạo các biểu đồ visualization
        if self.visualization_enabled:
            plot_paths = self._generate_visualizations()
            report_paths['plots'] = plot_paths
        
        # 3. Tạo báo cáo HTML
        html_path = self._generate_html_report()
        report_paths['html'] = html_path
        
        elapsed_time = time.time() - start_time
        logger.info(f"Hoàn thành báo cáo sau {elapsed_time:.2f} giây")
        
        return report_paths
    
    def _calculate_accuracy_by_model_prompt(self) -> pd.DataFrame:
        """
        Tính toán accuracy theo từng model và prompt type.
        
        Returns:
            pd.DataFrame: DataFrame chứa accuracy theo model và prompt
        """
        if 'is_correct' not in self.results_df.columns:
            return pd.DataFrame()
        
        # Xác định tên cột model để sử dụng
        model_col = 'model' if 'model' in self.results_df.columns else ('model_name' if 'model_name' in self.results_df.columns else None)
        
        if not model_col or 'prompt_type' not in self.results_df.columns:
            logger.warning(f"Không thể tính accuracy theo model/prompt: thiếu cột cần thiết. Có các cột: {list(self.results_df.columns)}")
            return pd.DataFrame()
        
        # Nhóm theo model và prompt_type, tính trung bình is_correct
        try:
            accuracy_df = self.results_df.groupby([model_col, 'prompt_type'])['is_correct'].mean().reset_index()
            accuracy_df.rename(columns={'is_correct': 'accuracy'}, inplace=True)
            
            # Đảm bảo cột 'model' luôn tồn tại
            if model_col == 'model_name':
                accuracy_df['model'] = accuracy_df['model_name']
            
            # Tính thêm số lượng mẫu
            counts = self.results_df.groupby([model_col, 'prompt_type']).size().reset_index(name='count')
            accuracy_df = accuracy_df.merge(counts, on=[model_col, 'prompt_type'])
            
            return accuracy_df
        except Exception as e:
            logger.error(f"Lỗi khi tính accuracy: {e}")
            return pd.DataFrame()
    
    def _calculate_metrics_by_model(self) -> pd.DataFrame:
        """
        Tính toán các metrics theo từng model.
        
        Returns:
            pd.DataFrame: DataFrame chứa các metrics theo model
        """
        metrics_df = pd.DataFrame()
        
        try:
            # Xác định tên cột model để sử dụng
            model_col = 'model' if 'model' in self.results_df.columns else ('model_name' if 'model_name' in self.results_df.columns else None)
            
            if not model_col:
                logger.warning(f"Không thể tính metrics theo model: thiếu cột model. Có các cột: {list(self.results_df.columns)}")
                return pd.DataFrame()
                
            metrics = []
            
            for model in self.results_df[model_col].unique():
                model_df = self.results_df[self.results_df[model_col] == model]
                model_metrics = {'model': model}
                
                # Accuracy
                if 'is_correct' in model_df.columns:
                    model_metrics['accuracy'] = model_df['is_correct'].mean()
                
                # Latency - kiểm tra cả hai tên cột có thể có
                latency_col = None
                for col_name in ['latency', 'elapsed_time']:
                    if col_name in model_df.columns:
                        latency_col = col_name
                        break
                        
                if latency_col:
                    model_metrics['avg_latency'] = model_df[latency_col].mean()
                    model_metrics['min_latency'] = model_df[latency_col].min()
                    model_metrics['max_latency'] = model_df[latency_col].max()
                
                # Token speed
                if 'tokens_per_second' in model_df.columns:
                    model_metrics['avg_tokens_per_second'] = model_df['tokens_per_second'].mean()
                
                # Response length
                if 'token_count' in model_df.columns:
                    model_metrics['avg_token_count'] = model_df['token_count'].mean()
                elif 'response_length' in model_df.columns:
                    model_metrics['avg_response_length'] = model_df['response_length'].mean()
                
                # Reasoning scores
                reasoning_cols = [col for col in model_df.columns if col.startswith('reasoning_') and col != 'reasoning_evaluation']
                for col in reasoning_cols:
                    if model_df[col].notna().any():
                        model_metrics[col] = model_df[col].mean()
                
                metrics.append(model_metrics)
            
            if metrics:
                metrics_df = pd.DataFrame(metrics)
                
            return metrics_df
        except Exception as e:
            logger.error(f"Lỗi khi tính metrics: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    def _init_default_template(self):
        """Khởi tạo template mặc định cho báo cáo HTML."""
        self.html_template = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <style>
        :root {
            --bg-color: {{ theme_vars.bg_color }};
            --text-color: {{ theme_vars.text_color }};
            --accent-color: {{ theme_vars.accent_color }};
            --border-color: {{ theme_vars.border_color }};
            --header-bg: {{ theme_vars.header_bg }};
            --card-bg: {{ theme_vars.card_bg }};
            --hover-color: {{ theme_vars.hover_color }};
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--bg-color);
            margin: 0;
            padding: 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background-color: var(--header-bg);
            color: #fff;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }
        
        h1, h2, h3, h4 {
            color: var(--accent-color);
            margin-top: 1.5em;
        }
        
        .card {
            background-color: var(--card-bg);
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        th {
            background-color: var(--header-bg);
            color: white;
        }
        
        tr:hover {
            background-color: var(--hover-color);
        }
        
        .plot-container {
            margin: 30px 0;
        }
        
        .plot-image {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
            border-radius: 5px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background-color: var(--card-bg);
            border-radius: 5px;
            padding: 20px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: var(--accent-color);
            margin: 10px 0;
        }
        
        .tab {
            overflow: hidden;
            border: 1px solid var(--border-color);
            background-color: var(--card-bg);
            border-radius: 5px 5px 0 0;
        }
        
        .tab button {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 17px;
            color: var(--text-color);
        }
        
        .tab button:hover {
            background-color: var(--hover-color);
        }
        
        .tab button.active {
            background-color: var(--accent-color);
            color: white;
        }
        
        .tabcontent {
            display: none;
            padding: 20px;
            border: 1px solid var(--border-color);
            border-top: none;
            border-radius: 0 0 5px 5px;
            animation: fadeEffect 1s;
        }
        
        @keyframes fadeEffect {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        
        footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            border-top: 1px solid var(--border-color);
            color: var(--text-color);
            opacity: 0.8;
        }
        
        .metrics-explanation {
            background-color: var(--card-bg);
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .metrics-explanation h3 {
            color: var(--accent-color);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }
        
        .metrics-explanation ul {
            padding-left: 20px;
        }
        
        .metrics-explanation li {
            margin-bottom: 10px;
        }
        
        .metrics-explanation strong {
            color: var(--accent-color);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ report_title }}</h1>
            <p>Báo cáo được tạo vào: {{ timestamp_formatted }}</p>
        </header>
        
        <div class="card">
            <h2>Giới Thiệu</h2>
            <p>Báo cáo này trình bày kết quả đánh giá các mô hình ngôn ngữ lớn (LLM) trên nhiều tiêu chí và prompt khác nhau. Mục đích của đánh giá này là so sánh hiệu suất của các mô hình trong việc trả lời câu hỏi, đánh giá chất lượng suy luận, và các khía cạnh khác của khả năng xử lý ngôn ngữ tự nhiên.</p>
        </div>
        
        <div class="metrics-explanation">
            <h3>Giải Thích Các Thước Đo Đánh Giá</h3>
            <p>Trong báo cáo này, chúng tôi sử dụng các thước đo đánh giá sau:</p>
            <ul>
                <li><strong>Accuracy (Độ chính xác):</strong> Tỷ lệ câu trả lời đúng trên tổng số câu hỏi. Đây là thước đo cơ bản nhất đánh giá khả năng trả lời chính xác của mô hình.</li>
                <li><strong>Latency (Độ trễ):</strong> Thời gian trung bình mô hình cần để sinh ra câu trả lời, tính bằng giây. Thước đo này giúp đánh giá hiệu suất thực tế của mô hình khi triển khai.</li>
                <li><strong>Completeness (Độ đầy đủ):</strong> Đánh giá mức độ đầy đủ thông tin trong câu trả lời của mô hình, với thang điểm từ 0 đến 1. Thước đo này đánh giá khả năng cung cấp toàn bộ thông tin cần thiết.</li>
                <li><strong>Consistency (Tính nhất quán):</strong> Đánh giá mức độ nhất quán giữa các câu trả lời khi đặt câu hỏi tương tự, với thang điểm từ 0 đến 1. Thước đo này giúp đánh giá độ tin cậy của mô hình.</li>
                <li><strong>Reasoning Quality (Chất lượng suy luận):</strong> Đánh giá chất lượng suy luận của mô hình dựa trên các tiêu chí như tính logic, tính chính xác về mặt toán học, sự rõ ràng, tính đầy đủ và tính liên quan. Mỗi tiêu chí được chấm điểm từ 1 đến 5.</li>
                <li><strong>Similarity (Độ tương đồng):</strong> Đo lường độ tương đồng giữa câu trả lời của mô hình và đáp án chuẩn, bao gồm ROUGE (đo trùng lặp n-gram), BLEU (đo độ chính xác n-gram) và Embedding Similarity (đo tương đồng ngữ nghĩa).</li>
            </ul>
        </div>
        
        <div class="card">
            <h2>Tổng Quan Đánh Giá</h2>
            
            <div class="summary-stats">
                {% for stat in summary_stats %}
                <div class="stat-card">
                    <div class="stat-title">{{ stat.title }}</div>
                    <div class="stat-value">{{ stat.value }}</div>
                    <div class="stat-desc">{{ stat.desc }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="tab">
            <button class="tablinks active" onclick="openTab(event, 'ModelComparison')">So Sánh Mô Hình</button>
            <button class="tablinks" onclick="openTab(event, 'PromptAnalysis')">Phân Tích Prompt</button>
            <button class="tablinks" onclick="openTab(event, 'ReasoningEval')">Đánh Giá Suy Luận</button>
            <button class="tablinks" onclick="openTab(event, 'DetailedResults')">Kết Quả Chi Tiết</button>
        </div>
        
        <div id="ModelComparison" class="tabcontent" style="display: block;">
            <h2>So Sánh Hiệu Suất Mô Hình</h2>
            
            {% if model_comparison_table %}
            <div class="card">
                <h3>Bảng So Sánh Mô Hình</h3>
                {{ model_comparison_table }}
            </div>
            {% endif %}
            
            {% if accuracy_plot_base64 %}
            <div class="plot-container">
                <h3>Biểu Đồ Accuracy Theo Mô Hình</h3>
                <img src="data:image/png;base64,{{ accuracy_plot_base64 }}" class="plot-image" alt="Accuracy Plot">
            </div>
            {% endif %}
            
            {% if latency_plot_base64 %}
            <div class="plot-container">
                <h3>Biểu Đồ Thời Gian Xử Lý Theo Mô Hình</h3>
                <img src="data:image/png;base64,{{ latency_plot_base64 }}" class="plot-image" alt="Latency Plot">
            </div>
            {% endif %}
            
            {% if metrics_boxplot_base64 %}
            <div class="plot-container">
                <h3>Phân Phối Các Metrics Theo Mô Hình</h3>
                <img src="data:image/png;base64,{{ metrics_boxplot_base64 }}" class="plot-image" alt="Metrics Boxplot">
            </div>
            {% endif %}
            
            {% if model_prompt_comparison_base64 %}
            <div class="plot-container">
                <h3>So Sánh Accuracy Giữa Các Mô Hình và Loại Prompt</h3>
                <img src="data:image/png;base64,{{ model_prompt_comparison_base64 }}" class="plot-image" alt="Model Prompt Comparison">
            </div>
            {% endif %}
            
            {% if model_prompt_count_base64 %}
            <div class="plot-container">
                <h3>Số Lượng Mẫu Theo Mô Hình và Loại Prompt</h3>
                <img src="data:image/png;base64,{{ model_prompt_count_base64 }}" class="plot-image" alt="Model Prompt Count">
            </div>
            {% endif %}
        </div>
        
        <div id="PromptAnalysis" class="tabcontent">
            <h2>Phân Tích Hiệu Quả Prompt</h2>
            
            {% if prompt_comparison_table %}
            <div class="card">
                <h3>Bảng So Sánh Các Loại Prompt</h3>
                {{ prompt_comparison_table }}
            </div>
            {% endif %}
            
            {% if prompt_accuracy_plot_base64 %}
            <div class="plot-container">
                <h3>Biểu Đồ Accuracy Theo Loại Prompt</h3>
                <img src="data:image/png;base64,{{ prompt_accuracy_plot_base64 }}" class="plot-image" alt="Prompt Accuracy Plot">
            </div>
            {% endif %}
            
            {% if heatmap_base64 %}
            <div class="plot-container">
                <h3>Heatmap Accuracy Theo Mô Hình và Prompt</h3>
                <img src="data:image/png;base64,{{ heatmap_base64 }}" class="plot-image" alt="Heatmap">
            </div>
            {% endif %}
            
            {% if sample_count_plot_base64 %}
            <div class="plot-container">
                <h3>Số Lượng Mẫu Theo Loại Prompt</h3>
                <img src="data:image/png;base64,{{ sample_count_plot_base64 }}" class="plot-image" alt="Sample Count Plot">
            </div>
            {% endif %}
        </div>
        
        <div id="ReasoningEval" class="tabcontent">
            <h2>Đánh Giá Khả Năng Suy Luận</h2>
            
            {% if reasoning_table %}
            <div class="card">
                <h3>Bảng Đánh Giá Suy Luận Theo Mô Hình</h3>
                {{ reasoning_table }}
            </div>
            {% endif %}
            
            {% if reasoning_plot_base64 %}
            <div class="plot-container">
                <h3>Biểu Đồ Đánh Giá Suy Luận</h3>
                <img src="data:image/png;base64,{{ reasoning_plot_base64 }}" class="plot-image" alt="Reasoning Evaluation Plot">
            </div>
            {% endif %}
            
            {% if radar_plot_base64 %}
            <div class="plot-container">
                <h3>Biểu Đồ Radar Chất Lượng Suy Luận</h3>
                <img src="data:image/png;base64,{{ radar_plot_base64 }}" class="plot-image" alt="Reasoning Radar Plot">
            </div>
            {% endif %}
            
            {% if completeness_plot_base64 %}
            <div class="plot-container">
                <h3>Đánh Giá Độ Đầy Đủ Của Câu Trả Lời</h3>
                <img src="data:image/png;base64,{{ completeness_plot_base64 }}" class="plot-image" alt="Completeness Plot">
            </div>
            {% endif %}
        </div>
        
        <div id="DetailedResults" class="tabcontent">
            <h2>Kết Quả Chi Tiết</h2>
            
            {% if sample_results_table %}
            <div class="card">
                <h3>Mẫu Kết Quả Đánh Giá</h3>
                {{ sample_results_table }}
            </div>
            {% endif %}
            
            {% if error_analysis_base64 %}
            <div class="plot-container">
                <h3>Phân Tích Lỗi</h3>
                <img src="data:image/png;base64,{{ error_analysis_base64 }}" class="plot-image" alt="Error Analysis Plot">
            </div>
            {% endif %}
            
            {% if consistency_plot_base64 %}
            <div class="plot-container">
                <h3>Đánh Giá Tính Nhất Quán</h3>
                <img src="data:image/png;base64,{{ consistency_plot_base64 }}" class="plot-image" alt="Consistency Plot">
            </div>
            {% endif %}
            
            {% if similarity_base64 %}
            <div class="plot-container">
                <h3>Đánh Giá Độ Tương Đồng</h3>
                <img src="data:image/png;base64,{{ similarity_base64 }}" class="plot-image" alt="Similarity Plot">
            </div>
            {% endif %}
            
            <div class="card">
                <p>Báo cáo đầy đủ có thể được tìm thấy trong các file CSV và JSON đi kèm.</p>
                <ul>
                    <li>CSV: <a href="{{ csv_path }}" target="_blank">{{ csv_filename }}</a></li>
                    <li>JSON: <a href="{{ json_path }}" target="_blank">{{ json_filename }}</a></li>
                </ul>
            </div>
        </div>
        
        <footer>
            <p>© {{ current_year }} LLM Evaluation Framework</p>
        </footer>
    </div>
    
    <script>
    function openTab(evt, tabName) {
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tabcontent");
        for (i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
        }
        tablinks = document.getElementsByClassName("tablinks");
        for (i = 0; i < tablinks.length; i++) {
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";
    }
    </script>
</body>
</html>
        """

    def _save_summary_csv(self) -> str:
        """
        Lưu tổng hợp kết quả phân tích dạng CSV.
        
        Returns:
            str: Đường dẫn đến file CSV
        """
        try:
            # Tạo DataFrame tổng hợp theo model và prompt_type
            summary_data = []
            
            # Nếu có dữ liệu accuracy theo model và prompt
            if not self.accuracy_by_model_prompt.empty:
                logger.info(f"Tạo báo cáo tóm tắt từ accuracy_by_model_prompt ({len(self.accuracy_by_model_prompt)} dòng)")
                for _, row in self.accuracy_by_model_prompt.iterrows():
                    model = row['model']
                    prompt_type = row['prompt_type']
                    model_prompt_df = self.results_df[(self.results_df['model'] == model) & 
                                                     (self.results_df['prompt_type'] == prompt_type)]
                    
                    summary_row = {
                        'model': model,
                        'prompt_type': prompt_type,
                        'accuracy': row['accuracy'],
                        'sample_count': row['count']
                    }
                    
                    # Thêm thông tin thời gian xử lý
                    if 'elapsed_time' in model_prompt_df.columns:
                        summary_row['avg_latency'] = model_prompt_df['elapsed_time'].mean()
                        summary_row['min_latency'] = model_prompt_df['elapsed_time'].min()
                        summary_row['max_latency'] = model_prompt_df['elapsed_time'].max()
                    
                    # Thêm thông tin token
                    if 'token_count' in model_prompt_df.columns:
                        summary_row['avg_tokens'] = model_prompt_df['token_count'].mean()
                    
                    # Thêm thêm các metrics khác nếu có
                    for col in model_prompt_df.columns:
                        if col.startswith('reasoning_') and col != 'reasoning_evaluation':
                            summary_row[col] = model_prompt_df[col].mean()
                        elif col in ['completeness_score', 'consistency_score', 'similarity_score']:
                            summary_row[col] = model_prompt_df[col].mean()
                    
                    summary_data.append(summary_row)
            else:
                # Tạo dữ liệu tóm tắt đơn giản nếu không có accuracy_by_model_prompt
                logger.info("Không có dữ liệu accuracy_by_model_prompt, tạo báo cáo từ results_df")
                
                # Kiểm tra nếu các cột cần thiết tồn tại
                model_col = 'model_name' if 'model_name' in self.results_df.columns else ('model' if 'model' in self.results_df.columns else None)
                
                if model_col and 'prompt_type' in self.results_df.columns:
                    # Nhóm dữ liệu theo model và prompt_type
                    grouped = self.results_df.groupby([model_col, 'prompt_type'])
                    
                    for (model, prompt_type), group in grouped:
                        summary_row = {
                            'model': model,
                            'prompt_type': prompt_type,
                            'sample_count': len(group)
                        }
                        
                        # Thêm accuracy nếu có
                        if 'is_correct' in group.columns:
                            summary_row['accuracy'] = group['is_correct'].mean()
                        
                        # Thêm thông tin thời gian xử lý
                        if 'elapsed_time' in group.columns:
                            summary_row['avg_latency'] = group['elapsed_time'].mean()
                            summary_row['min_latency'] = group['elapsed_time'].min()
                            summary_row['max_latency'] = group['elapsed_time'].max()
                        
                        # Thêm thông tin token
                        if 'token_count' in group.columns:
                            summary_row['avg_tokens'] = group['token_count'].mean()
                        
                        # Thêm thêm các metrics khác nếu có
                        for col in group.columns:
                            if col.startswith('reasoning_') and col != 'reasoning_evaluation':
                                summary_row[col] = group[col].mean()
                            elif col in ['completeness_score', 'consistency_score', 'similarity_score']:
                                summary_row[col] = group[col].mean()
                        
                        summary_data.append(summary_row)
                else:
                    # Tạo một summary rất đơn giản nếu thiếu các cột cơ bản
                    logger.warning("Thiếu các cột cơ bản cho báo cáo, tạo báo cáo tối thiểu")
                    
                    summary_row = {
                        'model': 'all_models',
                        'prompt_type': 'all_prompts',
                        'sample_count': len(self.results_df)
                    }
                    
                    # Thêm các thông số trung bình có thể tính được
                    for col in self.results_df.columns:
                        if col in ['is_correct', 'elapsed_time', 'token_count'] or \
                           col.startswith('reasoning_') or \
                           col in ['completeness_score', 'consistency_score', 'similarity_score']:
                            try:
                                if pd.api.types.is_numeric_dtype(self.results_df[col]):
                                    summary_row[f'avg_{col}'] = self.results_df[col].mean()
                            except:
                                pass
                    
                    summary_data.append(summary_row)
            
            # Tạo DataFrame và lưu vào CSV
            summary_df = pd.DataFrame(summary_data)
            
            # Đảm bảo thư mục đầu ra tồn tại
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Lưu vào CSV
            summary_df.to_csv(os.path.join(self.data_dir, f"evaluation_results_{self.timestamp}_analyzed.csv"), index=False)
            
            logger.info(f"Đã lưu báo cáo tóm tắt CSV tại: {os.path.join(self.data_dir, f'evaluation_results_{self.timestamp}_analyzed.csv')}")
            
            return os.path.join(self.data_dir, f"evaluation_results_{self.timestamp}_analyzed.csv")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu báo cáo tóm tắt CSV: {e}")
            logger.error(traceback.format_exc())
            return ""
    
    def _save_summary_json(self) -> str:
        """
        Lưu tổng hợp kết quả phân tích dạng JSON.
        
        Returns:
            str: Đường dẫn đến file JSON
        """
        try:
            # Tạo cấu trúc dữ liệu JSON
            json_data = {
                'report_info': {
                    'title': self.report_title,
                    'timestamp': self.timestamp,
                    'language': self.language
                },
                'overall_metrics': {},
                'models': {},
                'prompt_types': {},
                'model_prompt_combinations': []
            }
            
            # Thêm metrics tổng thể
            if 'is_correct' in self.results_df.columns:
                json_data['overall_metrics']['accuracy'] = float(self.results_df['is_correct'].mean())
                json_data['overall_metrics']['total_samples'] = len(self.results_df)
                json_data['overall_metrics']['correct_count'] = int(self.results_df['is_correct'].sum())
            
            if 'elapsed_time' in self.results_df.columns:
                json_data['overall_metrics']['avg_latency'] = float(self.results_df['elapsed_time'].mean())
            
            # Thêm metrics theo model
            for model in self.results_df['model'].unique():
                model_df = self.results_df[self.results_df['model'] == model]
                model_data = {
                    'sample_count': len(model_df)
                }
                
                # Accuracy
                if 'is_correct' in model_df.columns:
                    model_data['accuracy'] = float(model_df['is_correct'].mean())
                
                # Latency
                if 'elapsed_time' in model_df.columns:
                    model_data['avg_latency'] = float(model_df['elapsed_time'].mean())
                
                # Token metrics
                if 'token_count' in model_df.columns:
                    model_data['avg_token_count'] = float(model_df['token_count'].mean())
                
                if 'tokens_per_second' in model_df.columns:
                    model_data['avg_tokens_per_second'] = float(model_df['tokens_per_second'].mean())
                
                # Reasoning metrics
                reasoning_cols = [col for col in model_df.columns 
                                 if col.startswith('reasoning_') and col != 'reasoning_evaluation']
                
                if reasoning_cols:
                    model_data['reasoning_scores'] = {}
                    for col in reasoning_cols:
                        if model_df[col].notna().any():
                            criterion = col.replace('reasoning_', '')
                            model_data['reasoning_scores'][criterion] = float(model_df[col].mean())
                
                json_data['models'][model] = model_data
            
            # Thêm metrics theo prompt type
            for prompt_type in self.results_df['prompt_type'].unique():
                prompt_df = self.results_df[self.results_df['prompt_type'] == prompt_type]
                prompt_data = {
                    'sample_count': len(prompt_df)
                }
                
                # Accuracy
                if 'is_correct' in prompt_df.columns:
                    prompt_data['accuracy'] = float(prompt_df['is_correct'].mean())
                
                # Latency
                if 'elapsed_time' in prompt_df.columns:
                    prompt_data['avg_latency'] = float(prompt_df['elapsed_time'].mean())
                
                json_data['prompt_types'][prompt_type] = prompt_data
            
            # Thêm metrics theo từng tổ hợp model và prompt type
            if not self.accuracy_by_model_prompt.empty:
                for _, row in self.accuracy_by_model_prompt.iterrows():
                    model = row['model']
                    prompt_type = row['prompt_type']
                    combo_data = {
                        'model': model,
                        'prompt_type': prompt_type,
                        'accuracy': float(row['accuracy']),
                        'sample_count': int(row['count'])
                    }
                    
                    # Lấy thêm thông tin từ DataFrame chính
                    model_prompt_df = self.results_df[(self.results_df['model'] == model) & 
                                                     (self.results_df['prompt_type'] == prompt_type)]
                    
                    # Thêm thông tin thời gian xử lý
                    if 'elapsed_time' in model_prompt_df.columns:
                        combo_data['avg_latency'] = float(model_prompt_df['elapsed_time'].mean())
                    
                    # Thêm thông tin token nếu có
                    if 'token_count' in model_prompt_df.columns:
                        combo_data['avg_token_count'] = float(model_prompt_df['token_count'].mean())
                    
                    json_data['model_prompt_combinations'].append(combo_data)
            
            # Lưu vào file JSON
            with open(self.json_report_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Đã lưu báo cáo tổng hợp JSON tại: {self.json_report_path}")
            return self.json_report_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo JSON: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return ""

    def _generate_html_report(self) -> str:
        """
        Tạo báo cáo HTML với tất cả các kết quả phân tích và biểu đồ.
        
        Returns:
            str: Đường dẫn đến báo cáo HTML
        """
        try:
            # Chuẩn bị dữ liệu cho template HTML
            template_data = {
                'report_title': self.report_title,
                'timestamp': self.timestamp,
                'timestamp_formatted': self._format_timestamp(self.timestamp),
                'current_year': datetime.datetime.now().year,
                'theme_vars': self._get_theme_variables(),
                'summary_stats': self._prepare_summary_stats(),
                'csv_path': os.path.relpath(self.csv_report_path, self.reports_dir),
                'csv_filename': os.path.basename(self.csv_report_path),
                'json_path': os.path.relpath(self.json_report_path, self.reports_dir),
                'json_filename': os.path.basename(self.json_report_path),
            }
            
            # Chuẩn bị bảng so sánh model
            template_data['model_comparison_table'] = self._prepare_model_comparison_table()
            
            # Chuẩn bị bảng so sánh prompt
            template_data['prompt_comparison_table'] = self._prepare_prompt_comparison_table()
            
            # Chuẩn bị bảng đánh giá suy luận nếu có
            reasoning_cols = [col for col in self.results_df.columns if col.startswith('reasoning_') and col != 'reasoning_evaluation']
            if reasoning_cols:
                template_data['reasoning_table'] = self._prepare_reasoning_table()
            
            # Chuẩn bị bảng mẫu kết quả
            template_data['sample_results_table'] = self._prepare_sample_results_table()
            
            # Thêm các biểu đồ được mã hóa base64 nếu có
            plots_to_encode = {
                'accuracy_plot_base64': os.path.join(self.plots_dir, f"accuracy_by_model_{self.timestamp}.png"),
                'prompt_accuracy_plot_base64': os.path.join(self.plots_dir, f"accuracy_by_prompt_{self.timestamp}.png"),
                'heatmap_base64': os.path.join(self.plots_dir, f"accuracy_heatmap_{self.timestamp}.png"),
                'latency_plot_base64': os.path.join(self.plots_dir, f"latency_by_model_{self.timestamp}.png"),
                'reasoning_plot_base64': os.path.join(self.plots_dir, f"reasoning_evaluation_{self.timestamp}.png"),
                'radar_plot_base64': os.path.join(self.plots_dir, f"reasoning_radar_{self.timestamp}.png"),
                'sample_count_plot_base64': os.path.join(self.plots_dir, f"sample_count_{self.timestamp}.png"),
                # Thêm các biểu đồ khác đã được tạo nhưng chưa được đưa vào báo cáo
                'metrics_boxplot_base64': os.path.join(self.plots_dir, f"metrics_boxplot_{self.timestamp}.png"),
                'completeness_plot_base64': os.path.join(self.plots_dir, f"completeness_{self.timestamp}.png"),
                'consistency_plot_base64': os.path.join(self.plots_dir, f"consistency_{self.timestamp}.png"),
                'error_analysis_base64': os.path.join(self.plots_dir, f"error_analysis_{self.timestamp}.png"),
                'model_prompt_comparison_base64': os.path.join(self.plots_dir, f"model_prompt_comparison_{self.timestamp}.png"),
                'model_prompt_count_base64': os.path.join(self.plots_dir, f"model_prompt_count_{self.timestamp}.png"),
                'similarity_base64': os.path.join(self.plots_dir, f"similarity_{self.timestamp}.png")
            }
            
            for key, plot_path in plots_to_encode.items():
                if os.path.exists(plot_path):
                    template_data[key] = self._encode_image_base64(plot_path)
                    logger.debug(f"Encoded {plot_path} for HTML report")
                else:
                    logger.debug(f"Plot file not found: {plot_path}")
            
            # Render template với dữ liệu
            try:
                from jinja2 import Template
                template = Template(self.html_template)
                html_content = template.render(**template_data)
            except ImportError:
                # Fallback nếu không có Jinja2
                html_content = self._fallback_html_template(template_data)
            
            # Lưu vào file HTML
            with open(self.html_report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Đã tạo báo cáo HTML tại: {self.html_report_path}")
            return self.html_report_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo HTML: {e}")
            logger.error(traceback.format_exc())
            return ""
    
    def _prepare_summary_stats(self) -> List[Dict[str, str]]:
        """
        Chuẩn bị thống kê tóm tắt cho báo cáo.
        
        Returns:
            List[Dict[str, str]]: Danh sách các thống kê tóm tắt
        """
        stats = []
        
        # Thêm tổng số mẫu
        stats.append({
            'title': 'Tổng Số Mẫu',
            'value': str(len(self.results_df)),
            'desc': 'Số lượng câu hỏi đã đánh giá'
        })
        
        # Thêm số lượng mô hình
        stats.append({
            'title': 'Số Lượng Mô Hình',
            'value': str(len(self.results_df['model'].unique())),
            'desc': 'Số lượng mô hình đã đánh giá'
        })
        
        # Thêm số lượng loại prompt
        stats.append({
            'title': 'Số Loại Prompt',
            'value': str(len(self.results_df['prompt_type'].unique())),
            'desc': 'Số lượng loại prompt đã sử dụng'
        })
        
        # Thêm accuracy tổng thể nếu có
        if 'is_correct' in self.results_df.columns:
            overall_accuracy = self.results_df['is_correct'].mean()
            stats.append({
                'title': 'Accuracy Tổng Thể',
                'value': f"{overall_accuracy:.1%}",
                'desc': 'Tỷ lệ trả lời đúng trên tất cả mô hình và prompt'
            })
        
        # Thêm thời gian xử lý trung bình nếu có
        if 'elapsed_time' in self.results_df.columns:
            avg_time = self.results_df['elapsed_time'].mean()
            stats.append({
                'title': 'Thời Gian TB',
                'value': f"{avg_time:.2f}s",
                'desc': 'Thời gian xử lý trung bình'
            })
        
        # Thêm tính nhất quán nếu có
        if 'consistency_score' in self.results_df.columns:
            consistency = self.results_df['consistency_score'].mean()
            stats.append({
                'title': 'Tính Nhất Quán',
                'value': f"{consistency:.2f}",
                'desc': 'Điểm đánh giá tính nhất quán (0-1)'
            })
            
        # Thêm độ đầy đủ nếu có
        if 'completeness_score' in self.results_df.columns:
            completeness = self.results_df['completeness_score'].mean()
            stats.append({
                'title': 'Độ Đầy Đủ',
                'value': f"{completeness:.2f}",
                'desc': 'Điểm đánh giá độ đầy đủ của câu trả lời (0-1)'
            })
            
        # Thêm độ tương đồng semantic nếu có
        if 'embedding_similarity' in self.results_df.columns:
            similarity = self.results_df['embedding_similarity'].mean()
            stats.append({
                'title': 'Độ Tương Đồng',
                'value': f"{similarity:.2f}",
                'desc': 'Độ tương đồng với đáp án chuẩn (0-1)'
            })
        
        # Thêm mô hình tốt nhất nếu có thông tin accuracy
        if 'is_correct' in self.results_df.columns:
            best_model_acc = self.results_df.groupby('model')['is_correct'].mean()
            if not best_model_acc.empty:
                best_model = best_model_acc.idxmax()
                stats.append({
                    'title': 'Mô Hình Tốt Nhất',
                    'value': best_model,
                    'desc': f"Accuracy: {best_model_acc.max():.1%}"
                })
        
        # Thêm loại prompt tốt nhất nếu có thông tin accuracy
        if 'is_correct' in self.results_df.columns:
            best_prompt_acc = self.results_df.groupby('prompt_type')['is_correct'].mean()
            if not best_prompt_acc.empty:
                best_prompt = best_prompt_acc.idxmax()
                stats.append({
                    'title': 'Prompt Tốt Nhất',
                    'value': best_prompt,
                    'desc': f"Accuracy: {best_prompt_acc.max():.1%}"
                })
        
        return stats
    
    def _prepare_model_comparison_table(self) -> str:
        """
        Chuẩn bị bảng so sánh các mô hình.
        
        Returns:
            str: HTML của bảng so sánh
        """
        if self.metrics_by_model.empty:
            return ""
        
        # Chuẩn bị dữ liệu
        table_df = self.metrics_by_model.copy()
        
        # Định dạng các cột
        formatters = {}
        for col in table_df.columns:
            if col == 'model':
                continue
            elif 'accuracy' in col.lower():
                formatters[col] = lambda x: f"{x:.4f}"
            elif 'latency' in col.lower() or 'time' in col.lower():
                formatters[col] = lambda x: f"{x:.2f}s"
            elif any(term in col.lower() for term in ['token', 'count', 'length']):
                formatters[col] = lambda x: f"{x:.1f}"
            elif 'reasoning' in col.lower():
                formatters[col] = lambda x: f"{x:.2f}"
            else:
                formatters[col] = lambda x: f"{x}"
        
        # Chuyển DataFrame thành HTML table
        return table_df.to_html(formatters=formatters, classes='table table-striped', index=False)
    
    def _prepare_prompt_comparison_table(self) -> str:
        """
        Chuẩn bị bảng so sánh các loại prompt.
        
        Returns:
            str: HTML của bảng so sánh
        """
        if 'is_correct' not in self.results_df.columns:
            return ""
        
        try:
            # Tạo DataFrame so sánh các loại prompt
            prompt_metrics = []
            
            for prompt_type in self.results_df['prompt_type'].unique():
                prompt_df = self.results_df[self.results_df['prompt_type'] == prompt_type]
                
                prompt_row = {
                    'prompt_type': prompt_type,
                    'sample_count': len(prompt_df)
                }
                
                # Accuracy
                if 'is_correct' in prompt_df.columns:
                    prompt_row['accuracy'] = prompt_df['is_correct'].mean()
                
                # Latency
                if 'elapsed_time' in prompt_df.columns:
                    prompt_row['avg_latency'] = prompt_df['elapsed_time'].mean()
                
                # Token metrics
                if 'token_count' in prompt_df.columns:
                    prompt_row['avg_token_count'] = prompt_df['token_count'].mean()
                
                if 'tokens_per_second' in prompt_df.columns:
                    prompt_row['avg_tokens_per_second'] = prompt_df['tokens_per_second'].mean()
                
                prompt_metrics.append(prompt_row)
            
            if not prompt_metrics:
                return ""
                
            prompt_df = pd.DataFrame(prompt_metrics)
            
            # Định dạng các cột
            formatters = {}
            for col in prompt_df.columns:
                if col == 'prompt_type':
                    continue
                elif col == 'sample_count':
                    formatters[col] = lambda x: f"{int(x)}"
                elif 'accuracy' in col.lower():
                    formatters[col] = lambda x: f"{x:.4f}"
                elif 'latency' in col.lower() or 'time' in col.lower():
                    formatters[col] = lambda x: f"{x:.2f}s"
                elif any(term in col.lower() for term in ['token', 'count']):
                    formatters[col] = lambda x: f"{x:.1f}"
                else:
                    formatters[col] = lambda x: f"{x}"
            
            # Chuyển DataFrame thành HTML table
            return prompt_df.to_html(formatters=formatters, classes='table table-striped', index=False)
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo bảng so sánh prompt: {e}")
            return ""
    
    def _prepare_reasoning_table(self) -> str:
        """
        Chuẩn bị bảng đánh giá suy luận.
        
        Returns:
            str: HTML của bảng đánh giá suy luận
        """
        reasoning_cols = [col for col in self.results_df.columns if col.startswith('reasoning_') and col != 'reasoning_evaluation']
        
        if not reasoning_cols:
            return ""
        
        try:
            # Tạo DataFrame tổng hợp điểm suy luận theo model
            reasoning_data = []
            
            for model in self.results_df['model'].unique():
                model_df = self.results_df[self.results_df['model'] == model]
                
                model_scores = {'model': model}
                for col in reasoning_cols:
                    if model_df[col].notna().any():
                        criterion = col.replace('reasoning_', '')
                        model_scores[criterion] = model_df[col].mean()
                
                reasoning_data.append(model_scores)
            
            if not reasoning_data:
                return ""
                
            reasoning_df = pd.DataFrame(reasoning_data)
            
            # Định dạng các cột
            formatters = {col: lambda x: f"{x:.2f}" for col in reasoning_df.columns if col != 'model'}
            
            # Chuyển DataFrame thành HTML table
            return reasoning_df.to_html(formatters=formatters, classes='table table-striped', index=False)
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo bảng đánh giá suy luận: {e}")
            return ""
    
    def _prepare_sample_results_table(self) -> str:
        """
        Chuẩn bị bảng mẫu kết quả đánh giá.
        
        Returns:
            str: HTML của bảng mẫu kết quả
        """
        try:
            # Chọn một số mẫu kết quả để hiển thị
            sample_size = min(10, len(self.results_df))
            sample_df = self.results_df.sample(sample_size) if sample_size > 0 else self.results_df
            
            # Chọn và sắp xếp lại các cột cần hiển thị
            display_cols = ['model', 'prompt_type', 'question_text']
            
            # Thêm cột accuracy nếu có
            if 'is_correct' in sample_df.columns:
                display_cols.append('is_correct')
            
            # Thêm cột thời gian nếu có
            if 'elapsed_time' in sample_df.columns:
                display_cols.append('elapsed_time')
            
            # Thêm cột response preview nếu có
            if 'response' in sample_df.columns:
                # Tạo cột preview của response
                sample_df['response_preview'] = sample_df['response'].apply(
                    lambda x: (str(x)[:100] + '...') if len(str(x)) > 100 else str(x)
                )
                display_cols.append('response_preview')
            
            # Lọc các cột hiển thị
            display_df = sample_df[display_cols]
            
            # Định dạng các cột
            formatters = {}
            if 'elapsed_time' in display_cols:
                formatters['elapsed_time'] = lambda x: f"{x:.2f}s"
            if 'is_correct' in display_cols:
                formatters['is_correct'] = lambda x: 'Đúng' if x else 'Sai'
            
            # Chuyển DataFrame thành HTML table
            return display_df.to_html(formatters=formatters, classes='table table-striped', escape=False, index=False)
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo bảng mẫu kết quả: {e}")
            return ""
    
    def _get_theme_variables(self) -> Dict[str, str]:
        """
        Lấy các biến CSS dựa trên theme được chọn.
        
        Returns:
            Dict[str, str]: Dictionary chứa các biến CSS
        """
        # Theme sáng (mặc định)
        light_theme = {
            'bg_color': '#f5f7fa',
            'text_color': '#333',
            'accent_color': '#3498db',
            'border_color': '#ddd',
            'header_bg': '#2c3e50',
            'card_bg': '#fff',
            'hover_color': '#f0f0f0'
        }
        
        # Theme tối
        dark_theme = {
            'bg_color': '#1a1a1a',
            'text_color': '#f0f0f0',
            'accent_color': '#3498db',
            'border_color': '#444',
            'header_bg': '#2c3e50',
            'card_bg': '#333',
            'hover_color': '#444'
        }
        
        return dark_theme if self.theme.lower() == 'dark' else light_theme
    
    def _format_timestamp(self, timestamp: str) -> str:
        """
        Định dạng timestamp thành chuỗi ngày giờ đẹp hơn.
        
        Args:
            timestamp (str): Timestamp dạng chuỗi
            
        Returns:
            str: Timestamp đã định dạng
        """
        try:
            # Định dạng nếu timestamp là dạng YYYYmmdd_HHMMSS
            if re.match(r'\d{8}_\d{6}', timestamp):
                dt = datetime.datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                return dt.strftime('%d/%m/%Y %H:%M:%S')
            return timestamp
        except:
            return timestamp
    
    def _encode_image_base64(self, image_path: str) -> str:
        """
        Mã hóa ảnh dạng base64 để nhúng vào HTML.
        
        Args:
            image_path (str): Đường dẫn đến file ảnh
            
        Returns:
            str: Chuỗi base64 của ảnh
        """
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
        except Exception as e:
            logger.error(f"Lỗi khi mã hóa ảnh {image_path}: {e}")
            return ""
    
    def _fallback_html_template(self, data: Dict[str, Any]) -> str:
        """
        Template HTML đơn giản khi không có Jinja2.
        
        Args:
            data (Dict[str, Any]): Dữ liệu cho template
            
        Returns:
            str: HTML được tạo
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{data['report_title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{data['report_title']}</h1>
            <p>Báo cáo được tạo vào: {data['timestamp_formatted']}</p>
            
            <h2>Tổng Quan</h2>
        """
        
        # Thêm các thống kê tóm tắt
        html += "<ul>"
        for stat in data['summary_stats']:
            html += f"<li><b>{stat['title']}:</b> {stat['value']} ({stat['desc']})</li>"
        html += "</ul>"
        
        # Thêm bảng so sánh model nếu có
        if data.get('model_comparison_table'):
            html += "<h2>So Sánh Mô Hình</h2>"
            html += data['model_comparison_table']
        
        # Thêm bảng so sánh prompt nếu có
        if data.get('prompt_comparison_table'):
            html += "<h2>So Sánh Prompt</h2>"
            html += data['prompt_comparison_table']
        
        # Thêm bảng đánh giá suy luận nếu có
        if data.get('reasoning_table'):
            html += "<h2>Đánh Giá Suy Luận</h2>"
            html += data['reasoning_table']
        
        # Thêm thông tin file báo cáo
        html += """
            <h2>Báo Cáo Chi Tiết</h2>
            <p>Báo cáo đầy đủ có thể được tìm thấy trong các file CSV và JSON đi kèm:</p>
        """
        
        html += f"""
            <ul>
                <li>CSV: <a href="{data['csv_path']}">{data['csv_filename']}</a></li>
                <li>JSON: <a href="{data['json_path']}">{data['json_filename']}</a></li>
            </ul>
            
            <footer>
                <p>© {data['current_year']} LLM Evaluation Framework</p>
            </footer>
        </body>
        </html>
        """
        
        return html

    def _generate_visualizations(self) -> Dict[str, str]:
        """
        Tạo tất cả các biểu đồ visualization cho báo cáo.
        
        Returns:
            Dict[str, str]: Dictionary chứa đường dẫn đến các biểu đồ
        """
        if not self.visualization_enabled:
            return {}
        
        logger.info("Tạo các biểu đồ visualization...")
        plot_paths = {}
        
        try:
            # Danh sách biểu đồ cần tạo
            plot_functions = {
                'accuracy_by_model': self._create_accuracy_by_model_plot,
                'accuracy_by_prompt': self._create_accuracy_by_prompt_plot,
                'accuracy_heatmap': self._create_accuracy_heatmap,
                'latency_plot': self._create_latency_plot,
                'sample_count_plot': self._create_sample_count_plot,
                'reasoning_evaluation': self._create_reasoning_evaluation_plot,
                'reasoning_radar': self._create_reasoning_radar_plot,
                'error_analysis_plot': self._create_error_analysis_plot,
                'consistency_plot': self._create_consistency_plot,
                'completeness_plot': self._create_completeness_plot,
                'similarity_plot': self._create_similarity_plot,
                'metrics_boxplot': self._create_metrics_boxplot,
                'model_prompt_comparison': self._create_model_prompt_comparison_plot
            }
            
            # Tạo tất cả các biểu đồ và xử lý lỗi cho mỗi biểu đồ
            for plot_name, plot_function in plot_functions.items():
                try:
                    logger.debug(f"Đang tạo biểu đồ: {plot_name}")
                    plot_path = plot_function()
                    
                    # Nếu biểu đồ không được tạo, sử dụng fallback nếu có thể
                    if plot_path is None:
                        title = plot_name.replace('_', ' ').title()
                        plot_path = self._create_fallback_plot(title, "Không đủ dữ liệu")
                        
                    plot_paths[plot_name] = plot_path
                    
                except Exception as plot_error:
                    logger.error(f"Lỗi khi tạo biểu đồ {plot_name}: {str(plot_error)}")
                    logger.debug(traceback.format_exc())
                    
                    # Tạo biểu đồ thay thế cho lỗi
                    title = plot_name.replace('_', ' ').title()
                    plot_paths[plot_name] = self._create_fallback_plot(title, f"Lỗi: {str(plot_error)}")
                
            # Xác nhận số lượng biểu đồ đã tạo thành công
            successful_plots = sum(1 for path in plot_paths.values() if path is not None)
            logger.info(f"Đã tạo {successful_plots}/{len(plot_functions)} biểu đồ visualization")
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ: {str(e)}")
            logger.debug(traceback.format_exc())
        
        # Loại bỏ các đường dẫn None (nếu có lỗi khi tạo biểu đồ)
        plot_paths = {k: v for k, v in plot_paths.items() if v is not None}
        
        return plot_paths
    
    def _create_accuracy_by_model_plot(self) -> str:
        """
        Tạo biểu đồ hiển thị accuracy theo model.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'is_correct' not in self.results_df.columns:
            logger.info("Không thể tạo biểu đồ accuracy theo model: thiếu cột is_correct")
            return self._create_fallback_plot("Accuracy By Model", "Thiếu dữ liệu accuracy (is_correct)")
            
        try:
            # Xác định tên cột model để sử dụng
            model_col = 'model_name' if 'model_name' in self.results_df.columns else ('model' if 'model' in self.results_df.columns else None)
            
            if not model_col:
                logger.info(f"Không thể tạo biểu đồ accuracy theo model: thiếu cột model. Có các cột: {list(self.results_df.columns)}")
                return self._create_fallback_plot("Accuracy By Model", "Thiếu cột model_name/model")
                
            # Tính accuracy theo model
            accuracy_by_model = self.results_df.groupby(model_col)['is_correct'].mean().reset_index()
            
            if len(accuracy_by_model) == 0:
                logger.info("Không có dữ liệu accuracy theo model để tạo biểu đồ")
                return self._create_fallback_plot("Accuracy By Model", "Không có dữ liệu sau khi tính toán")
                
            # Tạo figure mới
            plt.figure(figsize=(10, 6))
            
            # Vẽ barplot
            sns.barplot(x=model_col, y='is_correct', data=accuracy_by_model, hue=model_col, legend=False)
            
            # Thêm nhãn % trên mỗi cột
            ax = plt.gca()
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                       f'{height:.1%}', ha="center", fontsize=12)
            
            # Thiết lập tiêu đề và nhãn
            plt.title('Accuracy theo Model', fontsize=15)
            plt.xlabel('Model', fontsize=13)
            plt.ylabel('Accuracy', fontsize=13)
            plt.ylim(0, 1.1)  # Giới hạn trục y từ 0 đến 1.1 để có chỗ cho nhãn
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"accuracy_by_model_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ accuracy theo model: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("Accuracy By Model", f"Lỗi: {str(e)}")
    
    def _create_accuracy_by_prompt_plot(self) -> str:
        """
        Tạo biểu đồ hiển thị accuracy theo loại prompt.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'is_correct' not in self.results_df.columns:
            return None
            
        try:
            # Kiểm tra cột prompt_type
            if 'prompt_type' not in self.results_df.columns:
                logger.warning(f"Không thể tạo biểu đồ accuracy theo prompt: thiếu cột prompt_type. Có các cột: {list(self.results_df.columns)}")
                return None
                
            # Tính accuracy theo prompt type
            accuracy_by_prompt = self.results_df.groupby('prompt_type')['is_correct'].mean().reset_index()
            
            if len(accuracy_by_prompt) == 0:
                logger.warning("Không có dữ liệu accuracy theo prompt để tạo biểu đồ")
                return None
                
            # Tạo figure mới
            plt.figure(figsize=(12, 6))
            
            # Vẽ barplot
            sns.barplot(x='prompt_type', y='is_correct', data=accuracy_by_prompt, hue='prompt_type', legend=False)
            
            # Thêm nhãn % trên mỗi cột
            ax = plt.gca()
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                       f'{height:.1%}', ha="center", fontsize=12)
            
            # Thiết lập tiêu đề và nhãn
            plt.title('Accuracy theo Loại Prompt', fontsize=15)
            plt.xlabel('Loại Prompt', fontsize=13)
            plt.ylabel('Accuracy', fontsize=13)
            plt.ylim(0, 1.1)  # Giới hạn trục y từ 0 đến 1.1 để có chỗ cho nhãn
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"accuracy_by_prompt_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ accuracy theo prompt: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    def _create_accuracy_heatmap(self) -> str:
        """
        Tạo biểu đồ heatmap hiển thị accuracy theo model và prompt type.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        try:
            # Xác định tên cột model để sử dụng
            model_col = 'model_name' if 'model_name' in self.results_df.columns else ('model' if 'model' in self.results_df.columns else None)
            
            if not model_col or 'prompt_type' not in self.results_df.columns:
                logger.info(f"Không thể tạo heatmap: thiếu cột cần thiết. Có các cột: {list(self.results_df.columns)}")
                # Tạo biểu đồ đơn giản thay thế nếu thiếu cột
                return self._create_fallback_plot("Accuracy Heatmap", "Thiếu cột model hoặc prompt_type")
            
            # Kiểm tra xem có đủ dữ liệu từ nhiều model và prompt không
            models = self.results_df[model_col].unique()
            prompts = self.results_df['prompt_type'].unique()
            
            if len(models) < 2 or len(prompts) < 2:
                logger.info(f"Dữ liệu không đủ đa dạng để tạo heatmap: {len(models)} model, {len(prompts)} prompt types")
                # Tạo biểu đồ đơn giản thay thế nếu dữ liệu không đủ đa dạng
                return self._create_simple_comparison_plot()
            
            # Tạo dữ liệu cho heatmap
            pivot_data = None
            
            # Nếu có cột is_correct, tạo heatmap accuracy
            if 'is_correct' in self.results_df.columns:
                # Tính accuracy cho mỗi cặp model/prompt
                pivot_data = self.results_df.pivot_table(
                    values='is_correct', 
                    index=model_col, 
                    columns='prompt_type', 
                    aggfunc='mean'
                )
            # Nếu không có cột is_correct, thử tạo heatmap dựa trên số lượng mẫu
            else:
                # Đếm số lượng mẫu cho mỗi cặp model/prompt
                pivot_data = self.results_df.pivot_table(
                    values='question_id', 
                    index=model_col, 
                    columns='prompt_type', 
                    aggfunc='count'
                )
            
            # Nếu pivot table rỗng hoặc chỉ có 1 dòng 1 cột, tạo dữ liệu mặc định
            if pivot_data is None or pivot_data.empty or len(pivot_data) < 1 or len(pivot_data.columns) < 1:
                # Tạo dữ liệu mặc định
                sample_data = {}
                for prompt in prompts:
                    sample_data[prompt] = [0] * len(models)
                
                pivot_data = pd.DataFrame(sample_data, index=models)
                
                # Thêm giá trị thực từ dữ liệu
                for model in models:
                    for prompt in prompts:
                        model_prompt_df = self.results_df[
                            (self.results_df[model_col] == model) & 
                            (self.results_df['prompt_type'] == prompt)
                        ]
                        if len(model_prompt_df) > 0:
                            if 'is_correct' in model_prompt_df.columns:
                                pivot_data.loc[model, prompt] = model_prompt_df['is_correct'].mean()
                            else:
                                pivot_data.loc[model, prompt] = len(model_prompt_df)
            
            # Tạo figure mới
            plt.figure(figsize=(14, 8))
            
            # Vẽ heatmap với annotation và cbar thích hợp
            if 'is_correct' in self.results_df.columns:
                # Heatmap accuracy
                sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.1%', 
                          linewidths=.5, cbar_kws={'label': 'Accuracy'})
                plt.title('Accuracy theo Model và Prompt Type', fontsize=16)
            else:
                # Heatmap số lượng mẫu
                sns.heatmap(pivot_data, annot=True, cmap='Blues', fmt='.0f', 
                          linewidths=.5, cbar_kws={'label': 'Số lượng mẫu'})
                plt.title('Số lượng mẫu theo Model và Prompt Type', fontsize=16)
            
            # Thiết lập nhãn
            plt.ylabel('Model', fontsize=14)
            plt.xlabel('Prompt Type', fontsize=14)
            plt.tight_layout()
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"accuracy_heatmap_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo heatmap so sánh model và prompt: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("Accuracy Heatmap", f"Lỗi: {str(e)}")
    
    def _create_simple_comparison_plot(self) -> str:
        """
        Tạo biểu đồ so sánh đơn giản khi không đủ dữ liệu cho heatmap.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        try:
            # Xác định tên cột model để sử dụng
            model_col = 'model_name' if 'model_name' in self.results_df.columns else ('model' if 'model' in self.results_df.columns else None)
            
            if not model_col or self.results_df.empty:
                return self._create_fallback_plot("So sánh Model và Prompt", "Không đủ dữ liệu")
                
            # Tạo figure mới
            plt.figure(figsize=(12, 8))
            
            # Nếu có cột is_correct, hiển thị accuracy
            if 'is_correct' in self.results_df.columns:
                if 'prompt_type' in self.results_df.columns:
                    # Vẽ barplot nhóm theo model và prompt
                    sns.barplot(x=model_col, y='is_correct', hue='prompt_type', data=self.results_df)
                    plt.title('Accuracy theo Model và Prompt Type', fontsize=16)
                    plt.ylabel('Accuracy', fontsize=14)
                else:
                    # Vẽ barplot chỉ theo model
                    sns.barplot(x=model_col, y='is_correct', data=self.results_df)
                    plt.title('Accuracy theo Model', fontsize=16)
                    plt.ylabel('Accuracy', fontsize=14)
            else:
                # Vẽ biểu đồ đếm số lượng mẫu
                if 'prompt_type' in self.results_df.columns:
                    # Đếm mẫu theo model và prompt
                    count_df = self.results_df.groupby([model_col, 'prompt_type']).size().reset_index(name='count')
                    sns.barplot(x=model_col, y='count', hue='prompt_type', data=count_df)
                    plt.title('Số lượng mẫu theo Model và Prompt Type', fontsize=16)
                    plt.ylabel('Số lượng mẫu', fontsize=14)
                else:
                    # Đếm mẫu chỉ theo model
                    count_df = self.results_df.groupby(model_col).size().reset_index(name='count')
                    sns.barplot(x=model_col, y='count', data=count_df)
                    plt.title('Số lượng mẫu theo Model', fontsize=16)
                    plt.ylabel('Số lượng mẫu', fontsize=14)
            
            # Thiết lập nhãn
            plt.xlabel('Model', fontsize=14)
            plt.tight_layout()
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"simple_comparison_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            logger.info(f"Đã tạo biểu đồ so sánh đơn giản thay thế cho heatmap")
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ so sánh đơn giản: {str(e)}")
            return self._create_fallback_plot("So sánh Model và Prompt", f"Lỗi: {str(e)}")
    
    def _create_latency_plot(self) -> str:
        """
        Tạo biểu đồ hiển thị thời gian xử lý theo model.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'elapsed_time' not in self.results_df.columns:
            logger.info("Cột elapsed_time không có trong dữ liệu, không thể tạo biểu đồ latency")
            return None
            
        try:
            # Tính thời gian trung bình theo model
            if 'model_name' in self.results_df.columns:
                # Xử lý bình thường nếu có cột model_name
                latency_by_model = self.results_df.groupby('model_name')['elapsed_time'].mean().reset_index()
            else:
                # Tạo DataFrame đơn giản nếu không có cột model_name
                latency_by_model = pd.DataFrame({
                    'model_name': ['model'],
                    'elapsed_time': [self.results_df['elapsed_time'].mean()]
                })
            
            # Tạo figure mới
            plt.figure(figsize=(10, 6))
            
            # Vẽ barplot
            if len(latency_by_model) > 0:
                sns.barplot(x='model_name', y='elapsed_time', data=latency_by_model, hue='model_name', legend=False)
                
                # Thêm nhãn trên mỗi cột
                ax = plt.gca()
                for i, p in enumerate(ax.patches):
                    height = p.get_height()
                    ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                           f'{height:.2f}s', ha="center", fontsize=12)
            else:
                # Trường hợp không có dữ liệu sau khi nhóm
                logger.info("Không có dữ liệu để tạo biểu đồ latency sau khi nhóm theo model")
                plt.text(0.5, 0.5, "Không đủ dữ liệu", ha='center', va='center', fontsize=14)
            
            # Thiết lập tiêu đề và nhãn
            plt.title('Thời gian xử lý trung bình theo Model', fontsize=15)
            plt.xlabel('Model', fontsize=13)
            plt.ylabel('Thời gian (giây)', fontsize=13)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"latency_by_model_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ latency: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Tạo biểu đồ đơn giản với thông báo lỗi
            try:
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"Lỗi khi tạo biểu đồ: {str(e)}", ha='center', va='center', fontsize=12, wrap=True)
                output_path = os.path.join(self.plots_dir, f"latency_by_model_{self.timestamp}.png")
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
                return output_path
            except:
                return None
    
    def _create_sample_count_plot(self) -> str:
        """
        Tạo biểu đồ hiển thị số lượng mẫu theo model và prompt type.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        try:
            # Đếm số lượng mẫu theo model và prompt type
            count_df = self.results_df.groupby(['model_name', 'prompt_type']).size().reset_index(name='count')
            
            if len(count_df) == 0:
                logger.warning("Không có dữ liệu mẫu để tạo biểu đồ")
                return None
                
            # Tạo figure mới
            plt.figure(figsize=(14, 8))
            
            # Vẽ barplot nhóm theo model và prompt type
            sns.barplot(x='model_name', y='count', hue='prompt_type', data=count_df)
            
            # Thiết lập tiêu đề và nhãn
            plt.title('Số lượng mẫu theo Model và Prompt Type', fontsize=15)
            plt.xlabel('Model', fontsize=13)
            plt.ylabel('Số lượng mẫu', fontsize=13)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Prompt Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"sample_count_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ số lượng mẫu: {str(e)}")
            return None
    
    def _create_reasoning_evaluation_plot(self) -> str:
        """
        Tạo biểu đồ đánh giá suy luận theo model.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        # Tìm các cột reasoning
        reasoning_cols = [col for col in self.results_df.columns 
                        if col.startswith('reasoning_') and col != 'reasoning_evaluation']
        
        if not reasoning_cols:
            return None
            
        try:
            # Tính điểm trung bình theo model
            reasoning_df = self.results_df.groupby('model_name')[reasoning_cols].mean().reset_index()
            
            if len(reasoning_df) == 0:
                logger.warning("Không có dữ liệu đánh giá suy luận để tạo biểu đồ")
                return None
                
            # Tạo figure mới
            plt.figure(figsize=(14, 8))
            
            # Chuẩn bị dữ liệu cho barplot
            melted_df = pd.melt(reasoning_df, id_vars=['model_name'], 
                              value_vars=[col for col in reasoning_cols],
                              var_name='Tiêu chí', value_name='Điểm')
            
            # Làm sạch tên tiêu chí
            melted_df['Tiêu chí'] = melted_df['Tiêu chí'].str.replace('reasoning_', '')
            
            # Vẽ barplot
            sns.barplot(x='model_name', y='Điểm', hue='Tiêu chí', data=melted_df)
            
            # Thiết lập tiêu đề và nhãn
            plt.title('Đánh giá khả năng suy luận theo Model', fontsize=15)
            plt.xlabel('Model', fontsize=13)
            plt.ylabel('Điểm trung bình (1-5)', fontsize=13)
            plt.ylim(0, 5.5)  # Giới hạn trục y
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Tiêu chí', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"reasoning_evaluation_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ đánh giá suy luận: {str(e)}")
            return None
    
    def _create_reasoning_radar_plot(self) -> str:
        """
        Tạo biểu đồ radar cho đánh giá suy luận theo model.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        # Tìm các cột reasoning
        reasoning_cols = [col for col in self.results_df.columns 
                        if col.startswith('reasoning_') and col != 'reasoning_evaluation']
        
        if not reasoning_cols:
            return None
            
        try:
            # Tính điểm trung bình theo model
            reasoning_df = self.results_df.groupby('model_name')[reasoning_cols].mean().reset_index()
            
            if len(reasoning_df) == 0 or len(reasoning_cols) < 3:
                logger.warning("Không đủ dữ liệu đánh giá suy luận để tạo biểu đồ radar")
                return None
                
            # Chuẩn bị dữ liệu cho radar plot
            models = reasoning_df['model_name'].values
            categories = [col.replace('reasoning_', '') for col in reasoning_cols]
            
            # Tạo figure mới
            plt.figure(figsize=(10, 10))
            
            # Tính toán các tham số cho radar plot
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Đóng radar plot
            
            # Tạo subplot
            ax = plt.subplot(111, polar=True)
            
            # Thiết lập các tham số
            plt.xticks(angles[:-1], categories, fontsize=12)
            ax.set_rlabel_position(0)
            plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=10)
            plt.ylim(0, 5)
            
            # Vẽ radar plot cho từng model
            for i, model in enumerate(models):
                values = reasoning_df.loc[reasoning_df['model_name'] == model, reasoning_cols].values.flatten().tolist()
                values += values[:1]  # Đóng radar plot
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
                ax.fill(angles, values, alpha=0.1)
            
            # Thêm legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Thiết lập tiêu đề
            plt.title('Đánh giá khả năng suy luận theo Model', fontsize=15, y=1.1)
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"reasoning_radar_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ radar đánh giá suy luận: {str(e)}")
            return None
    
    def _create_metrics_boxplot(self) -> str:
        """
        Tạo boxplot hiển thị phân phối các metrics theo model.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        try:
            # Lựa chọn các metrics để hiển thị trong boxplot
            valid_metrics = [
                ('elapsed_time', 'Thời gian xử lý (giây)'),
                ('tokens_per_second', 'Tokens/giây'),
                ('token_count', 'Số tokens')
            ]
            
            # Kiểm tra các metrics có sẵn
            metrics_to_plot = []
            for col, label in valid_metrics:
                if col in self.results_df.columns and self.results_df[col].notna().any():
                    metrics_to_plot.append((col, label))
            
            if not metrics_to_plot:
                logger.warning("Không có metrics phù hợp để tạo boxplot")
                return None
            
            # Tạo subplots cho mỗi metric
            fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(14, 6 * len(metrics_to_plot)))
            
            # Thêm padding cho figure để tránh lỗi tight_layout
            fig.tight_layout(pad=3.0)
            
            # Xử lý trường hợp chỉ có 1 metric
            if len(metrics_to_plot) == 1:
                axes = [axes]
            
            # Vẽ boxplot cho từng metric
            for i, (metric, label) in enumerate(metrics_to_plot):
                ax = axes[i]
                
                # Vẽ boxplot trên trục hiện tại
                sns.boxplot(x='model_name', y=metric, hue='model_name', data=self.results_df, ax=ax, palette='viridis', legend=False)
                
                # Chỉnh sửa trục
                ax.set_title(f'Phân phối {label} theo model', fontsize=14)
                ax.set_xlabel('Model', fontsize=12)
                ax.set_ylabel(label, fontsize=12)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Xoay nhãn nếu cần
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Điều chỉnh layout
            fig.subplots_adjust(hspace=0.5, bottom=0.15)
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"metrics_boxplot_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo boxplot metrics: {str(e)}")
            return None
    
    def _create_model_prompt_comparison_plot(self) -> str:
        """
        Tạo heatmap so sánh hiệu suất giữa các model và prompt types.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        try:
            if hasattr(self, 'accuracy_by_model_prompt') and not self.accuracy_by_model_prompt.empty:
                # Tạo pivot table để vẽ heatmap
                accuracy_pivot = self.accuracy_by_model_prompt.pivot_table(
                    index='model', columns='prompt_type', values='accuracy')
                
                # Thiết lập kích thước dựa trên số lượng mục
                height = max(6, len(accuracy_pivot) * 0.8)
                width = max(8, len(accuracy_pivot.columns) * 1.2)
                
                # Vẽ heatmap
                plt.figure(figsize=(width, height))
                
                # Sử dụng vmax=1.0 vì accuracy là 0-1
                ax = sns.heatmap(accuracy_pivot, annot=True, cmap='YlGnBu', vmin=0, vmax=1,
                               fmt='.3f', linewidths=.5, cbar_kws={'label': 'Accuracy'})
                
                # Thiết lập tiêu đề và nhãn
                plt.title('So sánh accuracy theo model và prompt type', fontsize=16)
                plt.tight_layout()
                
                # Lưu biểu đồ
                output_path = os.path.join(self.plots_dir, f"model_prompt_comparison_{self.timestamp}.png")
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                # Thêm biểu đồ so sánh count
                count_pivot = self.accuracy_by_model_prompt.pivot_table(
                    index='model', columns='prompt_type', values='count')
                
                plt.figure(figsize=(width, height))
                ax = sns.heatmap(count_pivot, annot=True, cmap='Greens', fmt='g',
                               linewidths=.5, cbar_kws={'label': 'Số mẫu'})
                
                # Thiết lập tiêu đề và nhãn
                plt.title('Số lượng mẫu theo model và prompt type', fontsize=16)
                plt.tight_layout()
                
                # Lưu biểu đồ
                count_path = os.path.join(self.plots_dir, f"model_prompt_count_{self.timestamp}.png")
                plt.savefig(count_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                return output_path
            else:
                # Tạo biểu đồ đơn giản khi không có đủ dữ liệu
                logger.info("Tạo biểu đồ đơn giản thay thế cho heatmap model-prompt do dữ liệu không đủ")
                # Tạo một biểu đồ đơn giản
                plt.figure(figsize=(8, 6))
                
                # Vẽ biểu đồ đơn giản với dữ liệu có sẵn
                model_col = 'model_name' if 'model_name' in self.results_df.columns else 'model'
                models = self.results_df[model_col].unique() if model_col in self.results_df.columns else ['Model 1']
                prompts = self.results_df['prompt_type'].unique() if 'prompt_type' in self.results_df.columns else ['Prompt 1']
                
                data = np.zeros((len(models), len(prompts)))
                
                # Đặt giá trị mẫu vào ma trận
                for i, model in enumerate(models):
                    for j, prompt in enumerate(prompts):
                        row_count = len(self.results_df[(self.results_df[model_col] == model) & 
                                                      (self.results_df['prompt_type'] == prompt)])
                        data[i, j] = row_count
                
                # Biến thành DataFrame
                df = pd.DataFrame(data, index=models, columns=prompts)
                
                # Vẽ heatmap
                sns.heatmap(df, annot=True, cmap='Blues', fmt='g', linewidths=.5, cbar_kws={'label': 'Số mẫu'})
                
                # Thiết lập tiêu đề và nhãn
                plt.title('Phân bố số lượng mẫu theo model và prompt type', fontsize=16)
                plt.ylabel('Model', fontsize=14)
                plt.xlabel('Prompt Type', fontsize=14)
                plt.tight_layout()
                
                # Lưu biểu đồ
                output_path = os.path.join(self.plots_dir, f"model_prompt_count_simple_{self.timestamp}.png")
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo plot so sánh model và prompt: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_fallback_plot("So sánh Model và Prompt", "Không đủ dữ liệu")

    def _create_consistency_plot(self) -> str:
        """
        Tạo biểu đồ hiển thị đánh giá tính nhất quán theo model.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'consistency_score' not in self.results_df.columns:
            return None
            
        try:
            # Tính điểm nhất quán trung bình theo model
            consistency_by_model = self.results_df.groupby('model_name')['consistency_score'].mean().reset_index()
            
            if len(consistency_by_model) == 0:
                logger.warning("Không có dữ liệu tính nhất quán để tạo biểu đồ")
                return None
                
            # Tạo figure mới
            plt.figure(figsize=(10, 6))
            
            # Vẽ barplot
            sns.barplot(x='model_name', y='consistency_score', data=consistency_by_model, hue='model_name', legend=False)
            
            # Thêm nhãn trên mỗi cột
            ax = plt.gca()
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha="center", fontsize=12)
            
            # Thiết lập tiêu đề và nhãn
            plt.title('Đánh giá tính nhất quán theo Model', fontsize=15)
            plt.xlabel('Model', fontsize=13)
            plt.ylabel('Điểm nhất quán (0-1)', fontsize=13)
            plt.ylim(0, 1.1)  # Giới hạn trục y
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"consistency_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ tính nhất quán: {str(e)}")
            return None
            
    def _create_completeness_plot(self) -> str:
        """
        Tạo biểu đồ hiển thị đánh giá tính đầy đủ theo model.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'completeness_score' not in self.results_df.columns:
            return None
            
        try:
            # Tính điểm đầy đủ trung bình theo model
            completeness_by_model = self.results_df.groupby('model_name')['completeness_score'].mean().reset_index()
            
            if len(completeness_by_model) == 0:
                logger.warning("Không có dữ liệu tính đầy đủ để tạo biểu đồ")
                return None
                
            # Tạo figure mới
            plt.figure(figsize=(10, 6))
            
            # Vẽ barplot
            sns.barplot(x='model_name', y='completeness_score', data=completeness_by_model, hue='model_name', legend=False)
            
            # Thêm nhãn trên mỗi cột
            ax = plt.gca()
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha="center", fontsize=12)
            
            # Thiết lập tiêu đề và nhãn
            plt.title('Đánh giá tính đầy đủ theo Model', fontsize=15)
            plt.xlabel('Model', fontsize=13)
            plt.ylabel('Điểm đầy đủ (0-1)', fontsize=13)
            plt.ylim(0, 1.1)  # Giới hạn trục y
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"completeness_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ tính đầy đủ: {str(e)}")
            return None
            
    def _create_similarity_plot(self) -> str:
        """
        Tạo biểu đồ hiển thị độ tương đồng theo model.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        # Kiểm tra các cột similarity
        similarity_cols = ['rouge_score', 'bleu_score', 'embedding_similarity']
        available_cols = [col for col in similarity_cols if col in self.results_df.columns]
        
        if not available_cols:
            return None
            
        try:
            # Tạo figure mới
            plt.figure(figsize=(12, 8))
            
            # Tính điểm trung bình và chuẩn bị dữ liệu
            similarity_data = []
            
            for col in available_cols:
                by_model = self.results_df.groupby('model_name')[col].mean().reset_index()
                by_model['metric'] = col.replace('_score', '').title() if '_score' in col else col.title()
                similarity_data.append(by_model)
            
            if not similarity_data:
                logger.warning("Không có dữ liệu độ tương đồng để tạo biểu đồ")
                return None
                
            # Gộp dữ liệu
            similarity_df = pd.concat(similarity_data)
            
            # Vẽ barplot nhóm theo model và metric
            sns.barplot(x='model_name', y=available_cols[0], hue='metric', data=similarity_df)
            
            # Thiết lập tiêu đề và nhãn
            plt.title('Đánh giá độ tương đồng theo Model', fontsize=15)
            plt.xlabel('Model', fontsize=13)
            plt.ylabel('Điểm tương đồng (0-1)', fontsize=13)
            plt.ylim(0, 1.1)  # Giới hạn trục y
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Metric')
            
            # Thêm padding cho figure để tránh lỗi tight_layout
            fig = plt.gcf()
            fig.subplots_adjust(right=0.85, bottom=0.15)
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"similarity_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ độ tương đồng: {str(e)}")
            return None
            
    def _create_error_analysis_plot(self) -> str:
        """
        Tạo biểu đồ phân tích lỗi theo model.
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        if 'error_type' not in self.results_df.columns:
            return None
            
        try:
            # Lọc dữ liệu lỗi (is_correct = False)
            error_df = self.results_df[self.results_df['is_correct'] == False].copy() if 'is_correct' in self.results_df.columns else self.results_df
            
            if len(error_df) == 0 or 'error_type' not in error_df.columns:
                logger.warning("Không có dữ liệu phân tích lỗi để tạo biểu đồ")
                return None
                
            # Đếm các loại lỗi theo model
            error_counts = error_df.groupby(['model_name', 'error_type']).size().reset_index(name='count')
            
            # Tạo figure mới
            plt.figure(figsize=(14, 8))
            
            # Vẽ barplot nhóm theo model và loại lỗi
            sns.barplot(x='model_name', y='count', hue='error_type', data=error_counts)
            
            # Thiết lập tiêu đề và nhãn
            plt.title('Phân tích lỗi theo Model', fontsize=15)
            plt.xlabel('Model', fontsize=13)
            plt.ylabel('Số lượng lỗi', fontsize=13)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Loại lỗi', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Điều chỉnh padding để tránh lỗi tight_layout
            fig = plt.gcf()
            fig.subplots_adjust(right=0.75, bottom=0.15)
            
            # Lưu biểu đồ
            output_path = os.path.join(self.plots_dir, f"error_analysis_{self.timestamp}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ phân tích lỗi: {str(e)}")
            return None

    def _create_fallback_plot(self, title: str, message: str = "Không đủ dữ liệu") -> str:
        """
        Tạo biểu đồ thay thế đơn giản khi không thể tạo biểu đồ chính.
        
        Args:
            title (str): Tiêu đề biểu đồ
            message (str): Thông báo hiển thị trong biểu đồ
            
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        try:
            # Tạo biểu đồ đơn giản
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)
            plt.title(title, fontsize=16)
            plt.axis('off')  # Ẩn trục
            
            # Tạo tên file dựa trên tiêu đề
            title_slug = title.lower().replace(' ', '_').replace('-', '_')
            output_path = os.path.join(self.plots_dir, f"fallback_{title_slug}_{self.timestamp}.png")
            
            # Lưu biểu đồ
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            logger.info(f"Đã tạo biểu đồ thay thế cho {title}")
            return output_path
        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ thay thế: {str(e)}")
            return None

# Hàm ở cấp module để tạo báo cáo từ file kết quả đã lưu
def generate_reports(results_file: str, output_dir: str, timestamp: str = None) -> Dict[str, str]:
    """
    Tạo báo cáo từ file kết quả đã lưu.
    
    Args:
        results_file (str): Đường dẫn đến file kết quả CSV
        output_dir (str): Thư mục đầu ra cho báo cáo (thư mục run)
        timestamp (str, optional): Mốc thời gian cho tên file báo cáo
        
    Returns:
        Dict[str, str]: Dictionary chứa đường dẫn đến các báo cáo được tạo
    """
    try:
        # Đọc dữ liệu kết quả
        logger.info(f"Đọc dữ liệu từ file kết quả: {results_file}")
        results_df = pd.read_csv(results_file)
        
        if results_df.empty:
            logger.warning("File kết quả không chứa dữ liệu")
            return {"error": "File kết quả trống"}
        
        # Kiểm tra cấu trúc dữ liệu cơ bản
        required_columns = ['model_name', 'prompt_type', 'question_id']
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        
        if missing_columns:
            logger.warning(f"Thiếu các cột sau trong dữ liệu: {missing_columns}")
            logger.info(f"Các cột có sẵn: {list(results_df.columns)}")
            
            # Thêm cột thiếu với giá trị mặc định
            for col in missing_columns:
                if col == 'model_name':
                    results_df[col] = 'unknown_model'
                elif col == 'prompt_type':
                    results_df[col] = 'unknown_prompt'
                elif col == 'question_id':
                    results_df[col] = results_df.index
        
        # Đảm bảo có ít nhất một dòng dữ liệu
        if len(results_df) == 0:
            logger.warning("Không có dữ liệu để tạo báo cáo sau khi lọc")
            return {"error": "Không có dữ liệu hợp lệ"}
        
        # Sử dụng timestamp từ tham số hoặc tạo mới
        if not timestamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Kiểm tra thư mục đầu ra
        try:
            # Đảm bảo thư mục tồn tại
            os.makedirs(output_dir, exist_ok=True)
            
            # Kiểm tra các thư mục con cần thiết
            required_subdirs = ['analyzed_results', 'reports', 'plots']
            for subdir in required_subdirs:
                subdir_path = os.path.join(output_dir, subdir)
                os.makedirs(subdir_path, exist_ok=True)
                
            logger.info(f"Đã kiểm tra và tạo cấu trúc thư mục đầu ra: {output_dir}")
        except PermissionError:
            logger.error(f"Không có quyền tạo thư mục đầu ra: {output_dir}")
            return {"error": f"Không có quyền tạo thư mục: {output_dir}"}
        except Exception as dir_error:
            logger.error(f"Lỗi khi tạo thư mục đầu ra: {str(dir_error)}")
            return {"error": f"Lỗi thư mục đầu ra: {str(dir_error)}"}
        
        # Tạo ReportGenerator và tạo báo cáo
        logger.info(f"Bắt đầu tạo báo cáo với {len(results_df)} kết quả")
        report_generator = ReportGenerator(
            results_df=results_df,
            output_dir=output_dir,
            timestamp=timestamp
        )
        
        # Thử tạo báo cáo
        try:
            report_paths = report_generator.generate_reports()
            logger.info(f"Đã tạo thành công {len(report_paths)} báo cáo")
            return report_paths
        except Exception as report_error:
            logger.error(f"Lỗi khi tạo báo cáo từ dữ liệu: {str(report_error)}")
            logger.debug(traceback.format_exc())
            
            # Thử tạo báo cáo tối thiểu nếu tạo báo cáo đầy đủ thất bại
            try:
                logger.info("Thử tạo báo cáo tối thiểu...")
                minimal_report_path = os.path.join(output_dir, "reports", f"minimal_report_{timestamp}.html")
                
                # Tạo báo cáo HTML đơn giản
                with open(minimal_report_path, 'w', encoding='utf-8') as f:
                    f.write(f"<html><head><title>Báo cáo đánh giá tối thiểu</title></head><body>")
                    f.write(f"<h1>Báo cáo đánh giá tối thiểu</h1>")
                    f.write(f"<p>Thời gian: {timestamp}</p>")
                    f.write(f"<p>Số lượng mẫu: {len(results_df)}</p>")
                    
                    # Thêm tóm tắt dữ liệu có sẵn
                    f.write(f"<h2>Tóm tắt dữ liệu</h2>")
                    
                    if 'model_name' in results_df.columns:
                        models = results_df['model_name'].unique()
                        f.write(f"<p>Models: {', '.join(models)}</p>")
                    
                    if 'prompt_type' in results_df.columns:
                        prompts = results_df['prompt_type'].unique()
                        f.write(f"<p>Prompts: {', '.join(prompts)}</p>")
                    
                    f.write(f"<p>Cột dữ liệu: {', '.join(results_df.columns)}</p>")
                    
                    # Thêm thông báo lỗi
                    f.write(f"<h2>Lỗi khi tạo báo cáo đầy đủ</h2>")
                    f.write(f"<p>{str(report_error)}</p>")
                    
                    f.write("</body></html>")
                
                logger.info(f"Đã tạo báo cáo tối thiểu: {minimal_report_path}")
                return {"minimal_report": minimal_report_path, "error": str(report_error)}
            except Exception as minimal_error:
                logger.error(f"Không thể tạo báo cáo tối thiểu: {str(minimal_error)}")
                return {"error": f"Không thể tạo bất kỳ báo cáo nào: {str(report_error)}"}
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo báo cáo: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}
