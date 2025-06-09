#!/bin/bash

# --- Git 大文件检查脚本 ---
# 功能：查找当前 HEAD 提交中最大的文件，并按降序排列。

echo "🔍  正在扫描当前分支 (HEAD) 中的大文件..."
echo "--------------------------------------------------"

# 使用 git ls-tree -r -l HEAD 来获取文件列表、blob 哈希、模式和大小（字节）
# -r: 递归地进入子目录
# -l: 显示对象大小
# HEAD: 检查当前最新的提交
#
# 输出格式示例:
# 100644 blob 2183b3e... 5335	path/to/file.txt
#
# 我们会对第四列（文件大小）进行反向数字排序 (-k4 -rn)
# awk 用于格式化输出，将字节转换为 MB 并美化显示
# head -n 20 只显示前 20 个最大的文件

git ls-tree -r -l HEAD | \
sort -k4 -rn | \
head -n 20 | \
awk '{
  size_bytes = $4;
  size_mb = size_bytes / 1024 / 1024;
  path = $5;
  for (i=6; i<=NF; i++) {
    path = path " " $i
  };
  printf "📦  %.2f MB\t%s\n", size_mb, path
}'

echo "--------------------------------------------------"
echo "✅  扫描完成。请检查以上列表，特别是大于 50MB 的文件。"
echo "💡  建议：在提交前清除 Jupyter Notebook (.ipynb) 的输出内容。"
