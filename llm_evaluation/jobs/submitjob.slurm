#!/bin/bash
#SBATCH --job-name=eval_trune         # Tên job hiển thị khi squeue
#SBATCH --output=logs_slurm/%x_%j.out # File log stdout (lưu tại thư mục logs_slurm)
#SBATCH --error=logs_slurm/%x_%j.err  # File log stderr (lưu lỗi tại đây)
#SBATCH --nodelist=node001            # Chạy đúng trên node001 như bạn yêu cầu
#SBATCH --gres=gpu:1                  # Yêu cầu 1 GPU
#SBATCH --cpus-per-task=8             # Yêu cầu 8 CPU core (tùy chỉnh nếu cần nhiều hơn)
#SBATCH --mem=64G                     # Yêu cầu 64GB RAM (tăng lên nếu model lớn bị OOM)
#SBATCH --time=72:00:00               # Thời gian chạy tối đa (72 giờ)

# Load conda environment
# --- Phần cấu hình môi trường ---
echo "Job started at $(date) on node $(hostname)"

# Dòng này quan trọng, nó thiết lập shell hiện tại để sử dụng conda
eval "$(conda shell.bash hook)"

# Bây giờ lệnh activate sẽ hoạt động
conda activate env_llm
echo "Successfully activated Conda environment: $CONDA_DEFAULT_ENV"

# Di chuyển đến đúng thư mục chứa main.py (Sử dụng đường dẫn tuyệt đối cho an toàn)
PROJECT_DIR=~/trune/CoT-tech/llm_evaluation
cd $PROJECT_DIR
echo "Working directory: $(pwd)"


# --- Phần chạy lệnh chính ---
# Bạn sửa các tham số chạy main.py ở dưới đây theo nhu cầu thực tế
# Ví dụ: chạy llama và gemini với zero_shot, giới hạn 50 câu hỏi để test

python main.py \
    --models qwen \
    --prompts zero_shot few_shot_3 few_shot_5 few_shot_7 cot cot_self_consistency_3 cot_self_consistency_5 cot_self_consistency_7 react \
    --max-questions 400 \
    --batch-size 10

echo "Job finished at $(date)"
