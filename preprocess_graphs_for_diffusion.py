import os
import torch
import glob
from tqdm import tqdm
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

"""
目的:
该脚本的快速（多线程）版本，用于为条件Diffusion模型准备数据。
它通过并行处理多个图文件来大幅提高速度，特别是在快速存储系统上。

工作流程:
1. 主线程扫描输入目录，获取所有待处理的图文件列表。
2. 创建一个线程池（ThreadPool），其工作线程数由用户指定。
3. 每个图文件的处理任务被提交到线程池中。
4. 每个工作线程执行以下操作：
   a. 加载一个图文件。
   b. 验证其所有 `latent_paths`。
   c. 如果有效，加载所有对应的latent文件。
   d. 将latents堆叠，附加到图对象上，并保存新的图文件。
5. 主线程收集所有工作线程的结果，更新进度条，并打印最终的总结报告。
"""

def process_single_graph(graph_path: str, output_dir: str) -> tuple[str, str]:
    """
    处理单个图文件。此函数将在一个独立的工作线程中执行。

    Args:
        graph_path (str): 单个原始图文件的路径。
        output_dir (str): 保存处理后文件的目录。

    Returns:
        tuple[str, str]: 一个元组，包含处理结果 ("success" 或 "skipped") 和文件名。
    """
    basename = os.path.basename(graph_path)
    try:
        graph_data = torch.load(graph_path, map_location='cpu')

        # --- 关键验证步骤 ---
        if not hasattr(graph_data, 'latent_paths'):
            # 对于多线程，我们返回状态而不是直接打印，以避免控制台输出混乱
            return "skipped_no_attr", basename

        latents_for_this_graph = []
        # 遍历所有节点的latent路径
        for node_idx, latent_path in enumerate(graph_data.latent_paths):
            if not latent_path or not os.path.exists(latent_path):
                return "skipped_bad_path", basename

            latent_tensor = torch.load(latent_path, map_location='cpu')
            latents_for_this_graph.append(latent_tensor)

        # --- 保存处理后的图 ---
        augmented_graph = graph_data.clone()
        augmented_graph.latent = torch.stack(latents_for_this_graph, dim=0).float()
        sample_id = basename.replace('_graph.pt', '')
        augmented_graph.sample_id = sample_id
        if hasattr(augmented_graph, 'latent_paths'):
            del augmented_graph.latent_paths
        
        output_path = os.path.join(output_dir, basename)
        torch.save(augmented_graph, output_path)
        
        return "success", basename

    except Exception as e:
        # 捕获任何意外错误
        print(f"\n错误：处理文件 '{basename}' 时发生意外错误: {e}")
        return "error", basename

def preprocess_for_diffusion_multithreaded(original_graph_dir: str, output_dir: str, num_workers: int):
    """
    使用多线程读取原始图，附加VAE latent张量，并保存为新的图文件。

    Args:
        original_graph_dir (str): 包含原始 `_graph.pt` 文件的目录路径。
        output_dir (str): 用于保存处理后的新图文件的目录路径。
        num_workers (int): 用于并行处理的线程数。
    """
    start_time = time.time()
    print("--- 开始为Diffusion模型进行快速（多线程）预处理 ---")
    print(f"源目录: {original_graph_dir}")
    print(f"输出目录: {output_dir}")
    print(f"使用的工作线程数: {num_workers}")

    os.makedirs(output_dir, exist_ok=True)

    all_graph_files = sorted(glob.glob(os.path.join(original_graph_dir, "*_graph.pt")))
    if not all_graph_files:
        print("错误：在源目录中未找到任何 `_graph.pt` 文件。")
        return

    # 初始化计数器
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # 使用ThreadPoolExecutor来管理线程池
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # `partial` 用于固定 `process_single_graph` 函数的 `output_dir` 参数
        # 这样 `executor.map` 就可以只迭代 `all_graph_files`
        worker_func = partial(process_single_graph, output_dir=output_dir)
        
        # 将所有任务提交给线程池，并使用tqdm来显示进度
        # `executor.map` 会将 `worker_func` 应用于 `all_graph_files` 中的每个元素
        future_to_graph = {executor.submit(worker_func, graph_file): graph_file for graph_file in all_graph_files}
        
        # 使用 as_completed 来获取已完成任务的结果，并更新进度条
        progress_bar = tqdm(as_completed(future_to_graph), total=len(all_graph_files), desc="正在并行处理图文件")
        
        for future in progress_bar:
            try:
                status, basename = future.result()
                if status == "success":
                    processed_count += 1
                elif status.startswith("skipped"):
                    skipped_count += 1
                else: # error
                    error_count += 1
            except Exception as exc:
                # 捕获工作函数本身可能抛出的异常
                graph_name = future_to_graph[future]
                print(f'\n文件 {graph_name} 在处理中生成了一个异常: {exc}')
                error_count += 1
    
    end_time = time.time()
    duration = end_time - start_time

    # --- 最终总结报告 ---
    print("\n" + "="*50)
    print("--- 预处理完成 ---")
    print(f"总耗时: {duration:.2f} 秒")
    print(f"成功处理并保存的图数量: {processed_count}")
    print(f"因数据无效而跳过的图数量: {skipped_count}")
    print(f"因错误而失败的图数量: {error_count}")
    print(f"分析的总图数量: {len(all_graph_files)}")
    print("="*50 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="为条件Diffusion模型快速（多线程）预处理图数据。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help="包含原始 `_graph.pt` 文件的目录路径。"
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help="用于保存处理后的新图文件的目录路径。"
    )
    parser.add_argument(
        '--num_workers', type=int, default=16,
        help="用于并行处理文件的线程数。对于I/O密集型任务，可以设置得比CPU核心数高。"
    )

    args = parser.parse_args()
    
    if args.num_workers <= 0:
        raise ValueError("`num_workers` 必须是正整数。")
        
    preprocess_for_diffusion_multithreaded(args.input_dir, args.output_dir, args.num_workers)