import os

def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # Chuyển đổi sang GB

cache_dir = "./cache/model"
if os.path.exists(cache_dir):
    models = os.listdir(cache_dir)
    print("Dung lượng các mô hình đã tải về:")
    for model in models:
        model_path = os.path.join(cache_dir, model)
        size_gb = get_folder_size(model_path)
        print(f"- {model}: {size_gb:.2f} GB")
else:
    print("Thư mục cache không tồn tại hoặc chưa có mô hình nào được tải về.")
