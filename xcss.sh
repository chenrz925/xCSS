 #!/bin/bash

# 检查输入参数
if [ -z "$1" ]; then
    echo "使用方法: $0 [-d] [-s SCENE_GRAPH] <场景根目录>"
    echo "选项:"
    echo "  -d            启用密集匹配模式"
    echo "  -s SCENE_GRAPH  设置场景图类型（默认：retrieval-100-25）"
    exit 1
fi

# 解析参数
DENSE_MATCHING=false
SCENE_GRAPH="retrieval-100-25"
while getopts "ds:" opt; do
    case $opt in
        d) DENSE_MATCHING=true ;;
        s) SCENE_GRAPH="$OPTARG" ;;
        *) exit 1 ;;
    esac
done
shift $((OPTIND -1))

set -eo pipefail

ROOT_DIR="$1"
export NERFSTUDIO_METHOD_CONFIGS="xcss=xcss.config:xcss"
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# 1. 生成匹配对（带错误检查）
if [ ! -f "$ROOT_DIR/pairs_src.txt" ]; then
    echo "开始生成匹配对..."
    python make_pairs.py \
        --dir "$ROOT_DIR/images/" \
        --output "$ROOT_DIR/pairs_src.txt" \
        --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
        --retrieval_model checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
        --scene_graph "$SCENE_GRAPH" || {
        echo "生成匹配对失败，请检查输入数据"
        exit 1
    }
else
    echo "检测到已存在匹配对文件，跳过生成步骤"
fi

# 2. 建图流程（带错误检查）
if [ -d "$ROOT_DIR/reconstruction/0" ] && [ -d "$ROOT_DIR/masks" ] && [ -d "$ROOT_DIR/depth" ]; then
    echo "检测到已有重建结果，跳过建图步骤"
else
    echo "开始建图流程..."
    DENSE_FLAG=""
    if [ "$DENSE_MATCHING" = true ]; then
        DENSE_FLAG="--dense_matching"
    fi

    python kapture_mast3r_mapping.py \
        --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
        --dir "$ROOT_DIR/images/" \
        --output "$ROOT_DIR/" \
        --pairsfile_path "$ROOT_DIR/pairs_src.txt" \
        $DENSE_FLAG || {
        echo "建图流程失败，请检查日志"
        exit 2
    }
fi

# 新增GCC版本检测逻辑
echo "搜索可用GCC版本（≤12）..."
declare -a gcc_versions
for v in {12..4}; do
    if command -v gcc-$v &> /dev/null; then
        gcc_versions+=("$v")
    elif command -v gcc$v &> /dev/null; then
        gcc_versions+=("$v")
    fi
done

if [ ${#gcc_versions[@]} -eq 0 ]; then
    echo "错误：未找到≤12的GCC版本，请先安装gcc<=12"
    exit 1
fi

# 选择最高版本
selected_version=$(printf "%s\n" "${gcc_versions[@]}" | sort -nr | head -n1)
echo "选择GCC版本：$selected_version"

# 设置环境变量
export CC=$(command -v gcc-$selected_version || command -v gcc$selected_version)
export CXX=$(command -v g++-$selected_version || command -v g++$selected_version)
echo "设置环境变量："
echo "CC=$CC"
echo "CXX=$CXX"

# 3. NeRF训练（带错误检查）
echo "开始NeRF训练..."
ns-train xcss \
    --data "$ROOT_DIR" || {
    echo "NeRF训练失败，请检查配置"
    exit 3
}