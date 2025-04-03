"""
Tiện ích trực quan hóa kết quả đánh giá mô hình LLM.
Cung cấp các hàm tạo biểu đồ, đồ thị và trực quan hóa dữ liệu.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes

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
                sns.barplot(x=model_col, y=latency_col, data=data, ax=ax)
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
    data: Dict[str, List[float]],
    criteria: List[str],
    title: str = 'Biểu đồ radar đánh giá',
    figsize: Tuple[int, int] = (8, 8),
    colors: Optional[Dict[str, str]] = None,
    fill_alpha: float = 0.1,
    max_value: float = 5.0
) -> Figure:
    """
    Tạo biểu đồ radar (spider chart) để so sánh nhiều metrics.
    
    Args:
        data: Dict với key là tên model và value là list các giá trị metrics
        criteria: Danh sách tên các tiêu chí đánh giá
        title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ
        colors: Dict ánh xạ giữa tên model và màu sắc
        fill_alpha: Độ mờ của vùng tô màu
        max_value: Giá trị tối đa trên trục radar
        
    Returns:
        Đối tượng Figure của matplotlib
    """
    try:
        # Kiểm tra dữ liệu
        if not data:
            raise ValueError("Dữ liệu trống")
        if not criteria:
            raise ValueError("Không có tiêu chí đánh giá")
        
        # Số lượng tiêu chí
        N = len(criteria)
        
        # Góc cho mỗi trục
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        
        # Đóng biểu đồ radar bằng cách lặp lại điểm đầu tiên
        angles += angles[:1]
        
        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=figsize, dpi=DPI_DEFAULT, subplot_kw=dict(polar=True))
        
        # Sử dụng màu mặc định nếu không chỉ định
        if colors is None:
            colors = {}
            color_list = list(COLORS.values())
            for i, model in enumerate(data.keys()):
                colors[model] = color_list[i % len(color_list)]
        
        # Vẽ cho từng model
        for i, (model, values) in enumerate(data.items()):
            # Đảm bảo data đóng vòng
            values_plot = values + values[:1]
            
            # Vẽ đường và tô màu
            color = colors.get(model, COLORS['primary'])
            ax.plot(angles, values_plot, 'o-', linewidth=2, label=model, color=color)
            ax.fill(angles, values_plot, alpha=fill_alpha, color=color)
        
        # Đặt tên cho các trục
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria)
        
        # Thiết lập giới hạn trục r
        ax.set_ylim(0, max_value)
        
        # Đặt tiêu đề
        ax.set_title(title, y=1.1)
        
        # Thêm legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        return fig
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ radar: {str(e)}")
        # Trả về figure trống trong trường hợp lỗi
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