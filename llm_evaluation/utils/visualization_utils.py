"""
Tiện ích trực quan hóa kết quả đánh giá mô hình LLM.
Cung cấp các hàm tạo biểu đồ, đồ thị và trực quan hóa dữ liệu.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import traceback

from .logging_utils import get_logger

logger = get_logger(__name__)

# Thiết lập style mặc định
plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE_DEFAULT = (10, 6)
DPI_DEFAULT = 100

# Màu sắc chung
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#bcbd22',
    'info': '#17becf',
    'dark': '#7f7f7f',
    'purple': '#9467bd',
    'pink': '#e377c2',
    'brown': '#8c564b'
}

def set_visualization_style(style: str = 'seaborn-v0_8-whitegrid', 
                           font_size: int = 10, 
                           use_dark_mode: bool = False):
    """
    Thiết lập style mặc định cho các biểu đồ.
    
    Args:
        style: Tên style matplotlib
        font_size: Kích thước font mặc định
        use_dark_mode: Sử dụng chế độ tối
    """
    try:
        plt.style.use(style)
        
        # Thiết lập kích thước font
        plt.rcParams.update({
            'font.size': font_size,
            'axes.titlesize': font_size + 2,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
            'legend.fontsize': font_size,
            'figure.titlesize': font_size + 4
        })
        
        # Thiết lập chế độ tối nếu cần
        if use_dark_mode:
            plt.rcParams.update({
                'figure.facecolor': '#2E3440',
                'axes.facecolor': '#2E3440',
                'axes.edgecolor': '#D8DEE9',
                'axes.labelcolor': '#E5E9F0',
                'text.color': '#E5E9F0',
                'xtick.color': '#E5E9F0',
                'ytick.color': '#E5E9F0',
                'grid.color': '#3B4252',
                'legend.facecolor': '#2E3440',
                'legend.edgecolor': '#D8DEE9',
                'savefig.facecolor': '#2E3440'
            })
        
        logger.info(f"Đã thiết lập style: {style}, font_size: {font_size}, dark_mode: {use_dark_mode}")
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập style: {str(e)}")

def create_accuracy_comparison_plot(
    data: pd.DataFrame,
    model_col: str = 'model',
    accuracy_col: str = 'accuracy',
    group_col: Optional[str] = None,
    title: str = 'So sánh độ chính xác giữa các mô hình',
    figsize: Tuple[int, int] = FIGSIZE_DEFAULT,
    color_map: Optional[Dict[str, str]] = None,
    sort_values: bool = True,
    horizontal: bool = False
) -> Figure:
    """
    Tạo biểu đồ so sánh độ chính xác giữa các mô hình.
    
    Args:
        data: DataFrame chứa dữ liệu
        model_col: Tên cột chứa tên mô hình
        accuracy_col: Tên cột chứa giá trị độ chính xác
        group_col: Tên cột nhóm (nếu có)
        title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ
        color_map: Dict ánh xạ giữa tên mô hình và màu sắc
        sort_values: Sắp xếp theo giá trị độ chính xác
        horizontal: Vẽ biểu đồ ngang thay vì dọc
        
    Returns:
        Đối tượng Figure của matplotlib
    """
    try:
        # Tạo bản sao dataframe để không ảnh hưởng đến dữ liệu gốc
        df = data.copy()
        
        # Kiểm tra dữ liệu
        if model_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột {model_col} trong dữ liệu")
        if accuracy_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột {accuracy_col} trong dữ liệu")
        if group_col and group_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột {group_col} trong dữ liệu")
        
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI_DEFAULT)
        
        # Xử lý theo nhóm nếu có group_col
        if group_col:
            # Nhóm và tính giá trị trung bình
            grouped_data = df.groupby([model_col, group_col])[accuracy_col].mean().reset_index()
            
            # Pivot để dễ vẽ biểu đồ
            pivot_data = grouped_data.pivot(index=model_col, columns=group_col, values=accuracy_col)
            
            # Sắp xếp theo giá trị trung bình nếu cần
            if sort_values:
                avg_values = pivot_data.mean(axis=1)
                pivot_data = pivot_data.loc[avg_values.sort_values().index]
            
            # Vẽ biểu đồ
            if horizontal:
                pivot_data.plot(kind='barh', ax=ax, figsize=figsize, legend=True)
            else:
                pivot_data.plot(kind='bar', ax=ax, figsize=figsize, legend=True)
            
            # Đặt tên các trục và tiêu đề
            if horizontal:
                ax.set_xlabel('Độ chính xác')
                ax.set_ylabel('Mô hình')
            else:
                ax.set_xlabel('Mô hình')
                ax.set_ylabel('Độ chính xác')
            
            ax.set_title(title)
            ax.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            
        else:
            # Nhóm và tính giá trị trung bình
            grouped_data = df.groupby(model_col)[accuracy_col].mean().reset_index()
            
            # Sắp xếp theo giá trị nếu cần
            if sort_values:
                grouped_data = grouped_data.sort_values(by=accuracy_col)
            
            # Thiết lập màu sắc nếu có
            colors = None
            if color_map:
                colors = [color_map.get(model, COLORS['primary']) for model in grouped_data[model_col]]
            
            # Vẽ biểu đồ
            if horizontal:
                bars = ax.barh(grouped_data[model_col], grouped_data[accuracy_col], color=colors)
                ax.set_xlabel('Độ chính xác')
                ax.set_ylabel('Mô hình')
            else:
                bars = ax.bar(grouped_data[model_col], grouped_data[accuracy_col], color=colors)
                ax.set_xlabel('Mô hình')
                ax.set_ylabel('Độ chính xác')
            
            # Thêm giá trị lên các cột
            for bar in bars:
                if horizontal:
                    value = bar.get_width()
                    pos_x = value + 0.01
                    pos_y = bar.get_y() + bar.get_height() / 2
                    ha, va = 'left', 'center'
                else:
                    value = bar.get_height()
                    pos_x = bar.get_x() + bar.get_width() / 2
                    pos_y = value + 0.01
                    ha, va = 'center', 'bottom'
                ax.text(pos_x, pos_y, f"{value:.2f}", ha=ha, va=va)
            
            ax.set_title(title)
        
        # Chỉnh sửa tỷ lệ trục y
        if not horizontal:
            ax.set_ylim(0, 1.1 * ax.get_ylim()[1])
        else:
            ax.set_xlim(0, 1.1 * ax.get_xlim()[1])
        
        # Đảm bảo biểu đồ vừa với kích thước figure
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ accuracy comparison: {str(e)}")
        # Trả về figure trống trong trường hợp lỗi
        return plt.figure()

def create_metric_heatmap(
    data: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    title: str = 'Heatmap',
    figsize: Tuple[int, int] = FIGSIZE_DEFAULT,
    cmap: str = 'YlGnBu',
    annot: bool = True,
    fmt: str = '.2f',
    center: Optional[float] = None
) -> Figure:
    """
    Tạo biểu đồ heatmap cho metric.
    
    Args:
        data: DataFrame chứa dữ liệu
        row_col: Tên cột làm hàng trong heatmap
        col_col: Tên cột làm cột trong heatmap
        value_col: Tên cột chứa giá trị
        title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ
        cmap: Bảng màu
        annot: Hiển thị giá trị trên heatmap
        fmt: Format cho giá trị
        center: Giá trị trung tâm cho bảng màu (nếu có)
        
    Returns:
        Đối tượng Figure của matplotlib
    """
    try:
        # Kiểm tra dữ liệu
        if row_col not in data.columns:
            raise ValueError(f"Không tìm thấy cột {row_col} trong dữ liệu")
        if col_col not in data.columns:
            raise ValueError(f"Không tìm thấy cột {col_col} trong dữ liệu")
        if value_col not in data.columns:
            raise ValueError(f"Không tìm thấy cột {value_col} trong dữ liệu")
        
        # Pivot dữ liệu để tạo heatmap
        pivot_data = data.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc='mean')
        
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI_DEFAULT)
        
        # Vẽ heatmap
        sns.heatmap(pivot_data, annot=annot, fmt=fmt, cmap=cmap, ax=ax, center=center, cbar_kws={'label': value_col})
        
        # Đặt tiêu đề
        ax.set_title(title)
        
        # Chỉnh sửa tỷ lệ figure
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ metric heatmap: {str(e)}")
        # Trả về figure trống trong trường hợp lỗi
        return plt.figure()

def create_latency_plot(
    data: pd.DataFrame,
    model_col: str = 'model',
    latency_col: str = 'latency',
    group_col: Optional[str] = None,
    title: str = 'Phân tích độ trễ',
    figsize: Tuple[int, int] = FIGSIZE_DEFAULT,
    plot_type: str = 'box',
    log_scale: bool = False
) -> Figure:
    """
    Tạo biểu đồ phân tích độ trễ.
    
    Args:
        data: DataFrame chứa dữ liệu
        model_col: Tên cột chứa tên mô hình
        latency_col: Tên cột chứa giá trị độ trễ
        group_col: Tên cột nhóm (nếu có)
        title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ
        plot_type: Loại biểu đồ ('box', 'violin', 'bar', 'scatter')
        log_scale: Sử dụng thang logarit cho trục y
        
    Returns:
        Đối tượng Figure của matplotlib
    """
    try:
        # Kiểm tra dữ liệu
        if model_col not in data.columns:
            raise ValueError(f"Không tìm thấy cột {model_col} trong dữ liệu")
        if latency_col not in data.columns:
            raise ValueError(f"Không tìm thấy cột {latency_col} trong dữ liệu")
        if group_col and group_col not in data.columns:
            raise ValueError(f"Không tìm thấy cột {group_col} trong dữ liệu")
        
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI_DEFAULT)
        
        # Xử lý theo nhóm
        if group_col:
            if plot_type == 'box':
                sns.boxplot(x=model_col, y=latency_col, hue=group_col, data=data, ax=ax)
            elif plot_type == 'violin':
                sns.violinplot(x=model_col, y=latency_col, hue=group_col, data=data, ax=ax)
            elif plot_type == 'bar':
                sns.barplot(x=model_col, y=latency_col, hue=group_col, data=data, ax=ax)
            elif plot_type == 'scatter':
                sns.stripplot(x=model_col, y=latency_col, hue=group_col, data=data, ax=ax, jitter=True, alpha=0.7)
            else:
                raise ValueError(f"Loại biểu đồ không hợp lệ: {plot_type}")
            
            # Điều chỉnh vị trí legend
            ax.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            if plot_type == 'box':
                sns.boxplot(x=model_col, y=latency_col, data=data, ax=ax)
            elif plot_type == 'violin':
                sns.violinplot(x=model_col, y=latency_col, data=data, ax=ax)
            elif plot_type == 'bar':
                sns.barplot(x=model_col, y=latency_col, hue=model_col, data=data, ax=ax, legend=False)
            elif plot_type == 'scatter':
                sns.stripplot(x=model_col, y=latency_col, data=data, ax=ax, jitter=True, alpha=0.7)
            else:
                raise ValueError(f"Loại biểu đồ không hợp lệ: {plot_type}")
        
        # Đặt tên các trục và tiêu đề
        ax.set_xlabel('Mô hình')
        ax.set_ylabel('Độ trễ (ms)')
        ax.set_title(title)
        
        # Sử dụng thang logarit nếu cần
        if log_scale:
            ax.set_yscale('log')
        
        # Thêm grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Đảm bảo biểu đồ vừa với kích thước figure
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ latency: {str(e)}")
        # Trả về figure trống trong trường hợp lỗi
        return plt.figure()

def create_radar_chart(
    data: Dict[str, Dict[str, float]],
    title: str = 'Model Performance Comparison',
    figsize: Tuple[int, int] = (10, 10),
    colors: Optional[List[str]] = None,
    min_value: float = 0,
    max_value: float = 5,
    show_legend: bool = True
) -> Figure:
    """
    Tạo biểu đồ radar (spiderweb) để so sánh hiệu suất của các model trên nhiều tiêu chí.
    
    Args:
        data: Dict với cấu trúc {model_name: {criteria1: score1, criteria2: score2, ...}}
        title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ
        colors: Danh sách màu sắc cho các model
        min_value: Giá trị tối thiểu trên thang đo
        max_value: Giá trị tối đa trên thang đo
        show_legend: Hiển thị legend
        
    Returns:
        Đối tượng Figure của matplotlib
    """
    try:
        # Kiểm tra dữ liệu
        if not data or not all(isinstance(v, dict) for v in data.values()):
            logger.warning("Dữ liệu không đúng định dạng")
            return plt.figure()
            
        # Lấy danh sách các tiêu chí (sẽ là các trục của radar chart)
        all_criteria = set()
        for model_data in data.values():
            all_criteria.update(model_data.keys())
        
        criteria = sorted(list(all_criteria))
        
        if not criteria:
            logger.warning("Không có tiêu chí nào để vẽ radar chart")
            return plt.figure()
        
        # Chuẩn hóa tên các tiêu chí
        criteria_labels = [c.replace('_', ' ').replace('reasoning ', '').title() for c in criteria]
        
        # Số lượng tiêu chí (số cạnh của radar chart)
        N = len(criteria)
        
        # Tạo góc cho mỗi trục
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        
        # Khép kín biểu đồ bằng cách lặp lại giá trị đầu tiên
        angles += angles[:1]
        
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI_DEFAULT, subplot_kw=dict(polar=True))
        
        # Nếu không có colors, tạo bảng màu mặc định
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
        
        # Vẽ radar chart cho mỗi model
        for i, (model_name, model_data) in enumerate(data.items()):
            # Lấy điểm cho tất cả các tiêu chí
            values = [model_data.get(c, 0) for c in criteria]
            
            # Khép kín biểu đồ bằng cách lặp lại giá trị đầu tiên
            values += values[:1]
            
            # Vẽ đường cho model
            color = colors[i % len(colors)] if isinstance(colors, list) else colors
            ax.plot(angles, values, 'o-', linewidth=2, color=color, label=model_name)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Thiết lập các tiêu chí làm nhãn
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria_labels)
        
        # Thiết lập giới hạn các trục
        ax.set_ylim(min_value, max_value)
        
        # Thêm grid
        ax.set_rgrids(np.arange(min_value, max_value+1, 1), angle=0, fontsize=8)
        
        # Thêm legend nếu cần
        if show_legend:
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Đặt tiêu đề
        ax.set_title(title, pad=20)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo radar chart: {str(e)}")
        traceback.print_exc()
        return plt.figure()

def create_prompt_type_comparison(
    data: pd.DataFrame,
    model_col: str = 'model',
    prompt_col: str = 'prompt_type',
    metric_col: str = 'accuracy',
    title: str = 'So sánh hiệu suất theo loại prompt',
    figsize: Tuple[int, int] = FIGSIZE_DEFAULT,
    plot_type: str = 'bar'
) -> Figure:
    """
    Tạo biểu đồ so sánh hiệu suất giữa các loại prompt.
    
    Args:
        data: DataFrame chứa dữ liệu
        model_col: Tên cột chứa tên mô hình
        prompt_col: Tên cột chứa loại prompt
        metric_col: Tên cột chứa metric cần so sánh
        title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ
        plot_type: Loại biểu đồ ('bar', 'line', 'point')
        
    Returns:
        Đối tượng Figure của matplotlib
    """
    try:
        # Kiểm tra dữ liệu
        if model_col not in data.columns:
            raise ValueError(f"Không tìm thấy cột {model_col} trong dữ liệu")
        if prompt_col not in data.columns:
            raise ValueError(f"Không tìm thấy cột {prompt_col} trong dữ liệu")
        if metric_col not in data.columns:
            raise ValueError(f"Không tìm thấy cột {metric_col} trong dữ liệu")
        
        # Nhóm dữ liệu
        grouped_data = data.groupby([model_col, prompt_col])[metric_col].mean().reset_index()
        
        # Pivot dữ liệu
        pivot_data = grouped_data.pivot(index=model_col, columns=prompt_col, values=metric_col)
        
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI_DEFAULT)
        
        # Vẽ biểu đồ
        if plot_type == 'bar':
            pivot_data.plot(kind='bar', ax=ax)
        elif plot_type == 'line':
            pivot_data.plot(kind='line', ax=ax, marker='o')
        elif plot_type == 'point':
            pivot_data.plot(kind='line', ax=ax, marker='o', linestyle='')
        else:
            raise ValueError(f"Loại biểu đồ không hợp lệ: {plot_type}")
        
        # Đặt tên các trục và tiêu đề
        ax.set_xlabel('Mô hình')
        ax.set_ylabel(metric_col)
        ax.set_title(title)
        
        # Điều chỉnh vị trí legend
        ax.legend(title=prompt_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Thêm grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Đảm bảo biểu đồ vừa với kích thước figure
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ prompt type comparison: {str(e)}")
        # Trả về figure trống trong trường hợp lỗi
        return plt.figure()

def create_sample_count_plot(
    data: pd.DataFrame,
    count_col: str,
    cat_col: Optional[str] = None,
    title: str = 'Phân bố số lượng mẫu',
    figsize: Tuple[int, int] = FIGSIZE_DEFAULT,
    plot_type: str = 'bar'
) -> Figure:
    """
    Tạo biểu đồ hiển thị số lượng mẫu.
    
    Args:
        data: DataFrame chứa dữ liệu
        count_col: Tên cột hoặc tên nhóm cần đếm
        cat_col: Tên cột phân loại (nếu có)
        title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ
        plot_type: Loại biểu đồ ('bar', 'pie')
        
    Returns:
        Đối tượng Figure của matplotlib
    """
    try:
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI_DEFAULT)
        
        # Đếm số lượng mẫu
        if cat_col:
            # Nếu có cột phân loại
            if cat_col not in data.columns:
                raise ValueError(f"Không tìm thấy cột {cat_col} trong dữ liệu")
            
            counts = data[cat_col].value_counts()
            
            if plot_type == 'bar':
                counts.plot(kind='bar', ax=ax)
                ax.set_xlabel(cat_col)
                ax.set_ylabel('Số lượng mẫu')
                
                # Thêm số lượng lên mỗi cột
                for i, v in enumerate(counts):
                    ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
                
            elif plot_type == 'pie':
                counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, shadow=False)
                ax.set_ylabel('')
            else:
                raise ValueError(f"Loại biểu đồ không hợp lệ: {plot_type}")
                
        else:
            # Nếu không có cột phân loại, đếm theo count_col
            if count_col not in data.columns:
                raise ValueError(f"Không tìm thấy cột {count_col} trong dữ liệu")
            
            # Vẽ histogram hoặc count plot
            if plot_type == 'bar':
                if pd.api.types.is_numeric_dtype(data[count_col]):
                    # Numeric column: histogram
                    sns.histplot(data=data, x=count_col, ax=ax, kde=True)
                    ax.set_xlabel(count_col)
                    ax.set_ylabel('Số lượng mẫu')
                else:
                    # Categorical column: count plot
                    sns.countplot(data=data, x=count_col, ax=ax)
                    ax.set_xlabel(count_col)
                    ax.set_ylabel('Số lượng mẫu')
                    
                    # Xoay nhãn nếu quá dài
                    plt.xticks(rotation=45, ha='right')
            
            elif plot_type == 'pie':
                counts = data[count_col].value_counts()
                counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, shadow=False)
                ax.set_ylabel('')
            else:
                raise ValueError(f"Loại biểu đồ không hợp lệ: {plot_type}")
        
        # Đặt tiêu đề
        ax.set_title(title)
        
        # Đảm bảo biểu đồ vừa với kích thước figure
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ sample count: {str(e)}")
        # Trả về figure trống trong trường hợp lỗi
        return plt.figure()

def save_figure(
    fig: Figure,
    output_path: str,
    dpi: int = 300,
    format: str = 'png',
    transparent: bool = False
) -> bool:
    """
    Lưu biểu đồ vào file.
    
    Args:
        fig: Đối tượng Figure
        output_path: Đường dẫn đầu ra
        dpi: Độ phân giải (dots per inch)
        format: Định dạng file ('png', 'pdf', 'svg', 'jpg')
        transparent: Nền trong suốt
        
    Returns:
        True nếu lưu thành công, False nếu thất bại
    """
    try:
        # Tạo thư mục đầu ra nếu cần
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Lưu figure
        fig.savefig(
            output_path,
            dpi=dpi,
            format=format,
            bbox_inches='tight',
            transparent=transparent
        )
        
        logger.info(f"Đã lưu biểu đồ vào: {output_path}")
        plt.close(fig)
        return True
    
    except Exception as e:
        logger.error(f"Lỗi khi lưu biểu đồ: {str(e)}")
        return False

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (8, 6),
    fmt: str = 'd',
    cmap: str = 'Blues',
    normalize: bool = False
) -> Figure:
    """
    Vẽ biểu đồ ma trận nhầm lẫn (confusion matrix).
    
    Args:
        cm: Ma trận nhầm lẫn
        class_names: Tên các classes
        title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ
        fmt: Format cho giá trị hiển thị ('d' cho số nguyên, '.2f' cho số thực)
        cmap: Bảng màu
        normalize: Chuẩn hóa giá trị (chia theo hàng)
        
    Returns:
        Đối tượng Figure của matplotlib
    """
    try:
        # Chuẩn hóa ma trận nếu cần
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI_DEFAULT)
        
        # Vẽ heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        
        # Thêm colorbar
        fig.colorbar(im, ax=ax)
        
        # Đặt tên các trục
        if class_names is not None:
            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(class_names)
        
        # Thêm giá trị lên mỗi ô
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # Đặt tên các trục và tiêu đề
        ax.set_xlabel('Dự đoán')
        ax.set_ylabel('Thực tế')
        ax.set_title(title)
        
        # Đảm bảo biểu đồ vừa với kích thước figure
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Lỗi khi vẽ confusion matrix: {str(e)}")
        # Trả về figure trống trong trường hợp lỗi
        return plt.figure()

def create_correlation_plot(
    data: pd.DataFrame,
    metrics_columns: List[str],
    title: str = 'Correlation Matrix of Metrics',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'coolwarm',
    annot: bool = True,
    fmt: str = '.2f',
    mask_upper: bool = True
) -> Figure:
    """
    Tạo biểu đồ tương quan (correlation matrix) giữa các metrics.
    
    Args:
        data: DataFrame chứa dữ liệu
        metrics_columns: Danh sách các cột metrics cần tính tương quan
        title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ
        cmap: Bảng màu
        annot: Hiển thị giá trị tương quan
        fmt: Format giá trị
        mask_upper: Ẩn nửa trên của ma trận (tránh hiển thị trùng lặp)
        
    Returns:
        Đối tượng Figure của matplotlib
    """
    try:
        # Tính tương quan giữa các cột
        available_cols = [col for col in metrics_columns if col in data.columns]
        if not available_cols:
            logger.warning("Không có cột metrics nào khả dụng để tính tương quan.")
            return plt.figure()
            
        # Chỉ lấy các hàng có giá trị
        subset_df = data[available_cols].dropna()
        
        if len(subset_df) < 5:
            logger.warning("Không đủ dữ liệu để tạo biểu đồ tương quan (cần ít nhất 5 hàng).")
            return plt.figure()
            
        corr_matrix = subset_df.corr()
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI_DEFAULT)
        
        # Tạo mask cho nửa trên nếu cần
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
        # Vẽ heatmap
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            cmap=cmap,
            annot=annot, 
            fmt=fmt,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            ax=ax
        )
        
        # Chỉnh sửa tên các trục
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        # Đặt tiêu đề
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ tương quan: {str(e)}")
        return plt.figure()

def create_error_distribution_chart(
    data: pd.DataFrame,
    error_type_col: str = 'error_type',
    group_by_col: str = 'model_name',
    title: str = 'Distribution of Error Types by Model',
    figsize: Tuple[int, int] = (12, 8),
    plot_type: str = 'stacked_bar',
    colors: Optional[List[str]] = None,
    normalize: bool = True
) -> Figure:
    """
    Tạo biểu đồ phân bố lỗi theo model hoặc prompt type.
    
    Args:
        data: DataFrame chứa dữ liệu
        error_type_col: Tên cột chứa loại lỗi
        group_by_col: Tên cột để nhóm (model_name hoặc prompt_type)
        title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ
        plot_type: Loại biểu đồ ('stacked_bar' hoặc 'pie')
        colors: Danh sách màu sắc
        normalize: Chuẩn hóa dữ liệu (%)
        
    Returns:
        Đối tượng Figure của matplotlib
    """
    try:
        # Kiểm tra dữ liệu
        if error_type_col not in data.columns:
            logger.warning(f"Không tìm thấy cột {error_type_col} trong dữ liệu")
            return plt.figure()
            
        if group_by_col not in data.columns:
            logger.warning(f"Không tìm thấy cột {group_by_col} trong dữ liệu")
            return plt.figure()
            
        # Lọc các hàng có thông tin lỗi
        error_df = data[data[error_type_col].notna()].copy()
        
        if len(error_df) == 0:
            logger.warning("Không có dữ liệu lỗi để phân tích")
            return plt.figure()
            
        # Thay thế các giá trị NaN hoặc rỗng bằng "Unknown"
        error_df[error_type_col] = error_df[error_type_col].fillna("Unknown")
        error_df.loc[error_df[error_type_col] == "", error_type_col] = "Unknown"
        
        # Tạo bảng tần suất
        error_counts = pd.crosstab(
            error_df[group_by_col], 
            error_df[error_type_col], 
            normalize=normalize
        )
        
        # Nếu normalize, chuyển sang phần trăm
        if normalize:
            error_counts = error_counts * 100
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI_DEFAULT)
        
        if plot_type == 'stacked_bar':
            # Biểu đồ cột chồng
            error_counts.plot(
                kind='bar', 
                stacked=True, 
                ax=ax, 
                colormap='tab20' if colors is None else None,
                color=colors
            )
            
            # Chỉnh sửa tên các trục
            ax.set_xlabel(group_by_col.replace('_', ' ').title())
            ax.set_ylabel('Percentage of Errors' if normalize else 'Count of Errors')
            ax.legend(title=error_type_col.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Thêm giá trị lên mỗi cột
            if normalize:
                for c in ax.containers:
                    labels = [f'{v:.1f}%' if v > 5 else '' for v in c.datavalues]
                    ax.bar_label(c, labels=labels, label_type='center')
            
        elif plot_type == 'pie':
            # Tạo biểu đồ tròn cho mỗi giá trị trong group_by_col
            nrows = int(np.ceil(len(error_counts) / 3))
            ncols = min(len(error_counts), 3)
            
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=DPI_DEFAULT)
            
            # Làm phẳng axes nếu cần
            if nrows == 1 and ncols == 1:
                axes = np.array([axes])
            elif nrows == 1 or ncols == 1:
                axes = axes.flatten()
                
            for i, (name, row) in enumerate(error_counts.iterrows()):
                if i < nrows * ncols:
                    # Tính chỉ số hàng, cột
                    if nrows == 1 and ncols == 1:
                        ax = axes[0]
                    elif nrows == 1 or ncols == 1:
                        ax = axes[i]
                    else:
                        ax = axes[i // ncols, i % ncols]
                    
                    # Vẽ biểu đồ tròn
                    wedges, texts, autotexts = ax.pie(
                        row, 
                        labels=None,
                        autopct='%1.1f%%' if normalize else '%d',
                        startangle=90,
                        colors=colors
                    )
                    
                    # Chỉnh màu chữ
                    for autotext in autotexts:
                        autotext.set_color('white')
                    
                    ax.set_title(name)
                    
            # Ẩn các axes không dùng
            for i in range(len(error_counts), nrows * ncols):
                if nrows == 1 and ncols == 1:
                    pass
                elif nrows == 1 or ncols == 1:
                    axes[i].axis('off')
                else:
                    axes[i // ncols, i % ncols].axis('off')
            
            # Thêm legend chung
            fig.legend(
                wedges, 
                error_counts.columns, 
                title=error_type_col.replace('_', ' ').title(),
                loc='lower center', 
                bbox_to_anchor=(0.5, 0)
            )
        
        # Đặt tiêu đề chung
        if plot_type == 'pie':
            fig.suptitle(title, fontsize=16)
        else:
            ax.set_title(title)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ phân bố lỗi: {str(e)}")
        traceback.print_exc()
        return plt.figure()

def create_reasoning_distribution_plot(
    data: pd.DataFrame,
    reasoning_cols: List[str],
    group_by_col: str = 'model_name',
    plot_type: str = 'box',
    title: str = 'Distribution of Reasoning Scores',
    figsize: Tuple[int, int] = (14, 10),
    palette: str = 'viridis'
) -> Figure:
    """
    Tạo biểu đồ phân phối điểm suy luận (reasoning scores).
    
    Args:
        data: DataFrame chứa dữ liệu
        reasoning_cols: Danh sách các cột điểm suy luận cần vẽ
        group_by_col: Tên cột để nhóm (model_name hoặc prompt_type)
        plot_type: Loại biểu đồ ('box', 'violin' hoặc 'swarm')
        title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ
        palette: Bảng màu seaborn
        
    Returns:
        Đối tượng Figure của matplotlib
    """
    try:
        # Kiểm tra dữ liệu
        available_cols = [col for col in reasoning_cols if col in data.columns]
        if not available_cols:
            logger.warning(f"Không tìm thấy cột điểm suy luận nào trong dữ liệu")
            return plt.figure()
            
        if group_by_col not in data.columns:
            logger.warning(f"Không tìm thấy cột {group_by_col} trong dữ liệu")
            return plt.figure()
        
        # Lọc dữ liệu có ít nhất một cột điểm
        reasoning_df = data[data[available_cols].notna().any(axis=1)].copy()
        
        if len(reasoning_df) == 0:
            logger.warning("Không có dữ liệu điểm suy luận để phân tích")
            return plt.figure()
        
        # Reshape dữ liệu sang dạng long format để dễ vẽ
        id_vars = [group_by_col]
        
        # Thêm các biến phân loại khác nếu có
        for col in ['prompt_type', 'question_type', 'difficulty']:
            if col in reasoning_df.columns and col != group_by_col:
                id_vars.append(col)
        
        # Chuyển sang dạng long format
        long_df = pd.melt(
            reasoning_df,
            id_vars=id_vars,
            value_vars=available_cols,
            var_name='reasoning_metric',
            value_name='score'
        )
        
        # Chuẩn hóa tên reasoning_metric để hiển thị đẹp hơn
        long_df['reasoning_metric'] = long_df['reasoning_metric'].apply(
            lambda x: x.replace('reasoning_', '').replace('_', ' ').title()
        )
        
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI_DEFAULT)
        
        # Vẽ biểu đồ dựa trên loại được chọn
        if plot_type == 'box':
            ax = sns.boxplot(
                data=long_df, 
                x='reasoning_metric', 
                y='score',
                hue=group_by_col,
                palette=palette,
                ax=ax
            )
            
        elif plot_type == 'violin':
            ax = sns.violinplot(
                data=long_df, 
                x='reasoning_metric', 
                y='score',
                hue=group_by_col,
                palette=palette,
                inner='quart',  # Hiển thị boxplot bên trong violin
                ax=ax
            )
            
        elif plot_type == 'swarm':
            ax = sns.boxplot(
                data=long_df, 
                x='reasoning_metric', 
                y='score',
                hue=group_by_col,
                palette=palette,
                ax=ax,
                width=0.6,
                fliersize=0  # Ẩn outliers từ boxplot
            )
            
            # Thêm swarmplot để thấy phân bố thực tế
            ax = sns.swarmplot(
                data=long_df, 
                x='reasoning_metric', 
                y='score',
                hue=group_by_col,
                palette=palette,
                alpha=0.6,
                dodge=True,
                ax=ax
            )
            
            # Tránh hiện 2 legend trùng nhau
            handles, labels = ax.get_legend_handles_labels()
            half = len(handles) // 2
            ax.legend(handles[:half], labels[:half], title=group_by_col)
        
        # Chỉnh sửa tên các trục
        ax.set_xlabel('Tiêu chí đánh giá suy luận')
        ax.set_ylabel('Điểm (1-5)')
        
        # Đặt tiêu đề
        ax.set_title(title)
        
        # Chỉnh legend nếu không phải swarm
        if plot_type != 'swarm':
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title=group_by_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ phân phối điểm suy luận: {str(e)}")
        traceback.print_exc()
        return plt.figure()

def create_interactive_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = 'Interactive Bar Chart',
    height: int = 600,
    color_discrete_map: Optional[Dict[str, str]] = None
) -> 'plotly.graph_objects.Figure':
    """
    Tạo biểu đồ cột tương tác bằng Plotly.
    
    Args:
        data: DataFrame chứa dữ liệu
        x_col: Tên cột làm trục x
        y_col: Tên cột làm trục y
        color_col: Tên cột để phân biệt màu sắc (nếu có)
        title: Tiêu đề biểu đồ
        height: Chiều cao biểu đồ (px)
        color_discrete_map: Dict ánh xạ giá trị trong color_col với mã màu
        
    Returns:
        Đối tượng Figure của plotly
    """
    try:
        # Import plotly
        import plotly.express as px
        
        # Kiểm tra dữ liệu
        if x_col not in data.columns:
            logger.warning(f"Không tìm thấy cột {x_col} trong dữ liệu")
            return px.bar()
            
        if y_col not in data.columns:
            logger.warning(f"Không tìm thấy cột {y_col} trong dữ liệu")
            return px.bar()
            
        if color_col and color_col not in data.columns:
            logger.warning(f"Không tìm thấy cột {color_col} trong dữ liệu")
            color_col = None
        
        # Tạo biểu đồ
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            height=height,
            color_discrete_map=color_discrete_map,
            template="plotly_white",
            hover_data=data.columns,  # Hiển thị tất cả dữ liệu khi hover
            text=y_col  # Hiển thị giá trị trên mỗi cột
        )
        
        # Chỉnh định dạng
        fig.update_traces(
            texttemplate='%{text:.2f}', 
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return fig
        
    except ImportError:
        logger.warning("Không thể tạo biểu đồ tương tác. Vui lòng cài đặt plotly: pip install plotly")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ cột tương tác: {str(e)}")
        traceback.print_exc()
        import plotly.express as px
        return px.bar()

def create_interactive_heatmap(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    title: str = 'Interactive Heatmap',
    height: int = 600,
    colorscale: str = 'Viridis'
) -> 'plotly.graph_objects.Figure':
    """
    Tạo biểu đồ heatmap tương tác bằng Plotly.
    
    Args:
        data: DataFrame chứa dữ liệu
        x_col: Tên cột làm trục x
        y_col: Tên cột làm trục y
        value_col: Tên cột chứa giá trị
        title: Tiêu đề biểu đồ
        height: Chiều cao biểu đồ (px)
        colorscale: Bảng màu
        
    Returns:
        Đối tượng Figure của plotly
    """
    try:
        # Import plotly
        import plotly.express as px
        
        # Kiểm tra dữ liệu
        if x_col not in data.columns:
            logger.warning(f"Không tìm thấy cột {x_col} trong dữ liệu")
            return px.imshow()
            
        if y_col not in data.columns:
            logger.warning(f"Không tìm thấy cột {y_col} trong dữ liệu")
            return px.imshow()
            
        if value_col not in data.columns:
            logger.warning(f"Không tìm thấy cột {value_col} trong dữ liệu")
            return px.imshow()
        
        # Pivot dữ liệu
        pivot_df = data.pivot_table(
            index=y_col, 
            columns=x_col, 
            values=value_col,
            aggfunc='mean'
        )
        
        # Tạo biểu đồ
        fig = px.imshow(
            pivot_df,
            title=title,
            height=height,
            color_continuous_scale=colorscale,
            text_auto='.2f',  # Hiển thị giá trị trong mỗi ô
            aspect='auto'     # Tự động điều chỉnh tỷ lệ
        )
        
        # Chỉnh định dạng
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return fig
        
    except ImportError:
        logger.warning("Không thể tạo biểu đồ tương tác. Vui lòng cài đặt plotly: pip install plotly")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ heatmap tương tác: {str(e)}")
        traceback.print_exc()
        import plotly.express as px
        return px.imshow()

def create_interactive_scatter_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    text_col: Optional[str] = None,
    title: str = 'Interactive Scatter Plot',
    height: int = 600,
    opacity: float = 0.7
) -> 'plotly.graph_objects.Figure':
    """
    Tạo biểu đồ scatter tương tác bằng Plotly.
    
    Args:
        data: DataFrame chứa dữ liệu
        x_col: Tên cột làm trục x
        y_col: Tên cột làm trục y
        color_col: Tên cột để phân biệt màu sắc (nếu có)
        size_col: Tên cột để điều chỉnh kích thước điểm (nếu có)
        text_col: Tên cột hiển thị khi hover (nếu có)
        title: Tiêu đề biểu đồ
        height: Chiều cao biểu đồ (px)
        opacity: Độ trong suốt của điểm (0-1)
        
    Returns:
        Đối tượng Figure của plotly
    """
    try:
        # Import plotly
        import plotly.express as px
        
        # Kiểm tra dữ liệu
        if x_col not in data.columns:
            logger.warning(f"Không tìm thấy cột {x_col} trong dữ liệu")
            return px.scatter()
            
        if y_col not in data.columns:
            logger.warning(f"Không tìm thấy cột {y_col} trong dữ liệu")
            return px.scatter()
        
        # Chuẩn bị hover_data
        hover_data = []
        for col in data.columns:
            if col not in [x_col, y_col, color_col, size_col] and data[col].dtype != 'object':
                hover_data.append(col)
        
        # Tạo biểu đồ
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            text=text_col,
            title=title,
            height=height,
            opacity=opacity,
            template="plotly_white",
            hover_data=hover_data,
            trendline="ols" if data[x_col].dtype != 'object' and data[y_col].dtype != 'object' else None
        )
        
        # Chỉnh định dạng
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return fig
        
    except ImportError:
        logger.warning("Không thể tạo biểu đồ tương tác. Vui lòng cài đặt plotly: pip install plotly")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ scatter tương tác: {str(e)}")
        traceback.print_exc()
        import plotly.express as px
        return px.scatter() 