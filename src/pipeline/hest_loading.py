# 文件：hest_reader.py

import os
import glob
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import openslide
from typing import Optional, Union, List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

class HESTSample:
    """
    表示单个 HEST 样本，包括:
      - Sample ID (如 'TENX95')
      - AnnData 对象 (空间转录组数据)
      - Whole Slide Image (WSI) 对象
      - 其他可选文件，如 patches, transcripts 等
    支持懒加载和全加载模式。
    """

    def __init__(
        self,
        sample_id: str,
        st_path: str,
        wsi_path: str,
        patches_dir: Optional[str] = None,
        transcripts_path: Optional[str] = None,
        metadata_dict: Optional[Dict] = None,
        spatial_plot_path: Optional[str] = None,
        load_adata: bool = True,
        adata_lazy: bool = True,
        load_wsi: bool = True
    ):
        """
        初始化样本对象。

        参数:
          sample_id (str): 样本 ID。
          st_path (str): .h5ad 文件路径。
          wsi_path (str): WSI 文件路径。
          patches_dir (Optional[str]): patches 文件夹路径。
          transcripts_path (Optional[str]): transcripts 文件路径。
          metadata_dict (Optional[Dict]): 元数据字典。
          spatial_plot_path (Optional[str]): 预生成的空间转录组图像路径。
          load_adata (bool): 是否加载 AnnData 对象。
          adata_lazy (bool): 如果加载 AnnData，是否使用懒加载模式。
          load_wsi (bool): 是否加载 WSI 图像。
        """
        self.sample_id = sample_id
        self.st_path = st_path
        self.wsi_path = wsi_path
        self.patches_dir = patches_dir
        self.transcripts_path = transcripts_path
        self.metadata_dict = metadata_dict if metadata_dict else {}
        self.spatial_plot_path = spatial_plot_path  # 预生成的空间转录组图路径

        self.adata = None  # AnnData 对象
        self.wsi = None    # OpenSlide 对象

        if load_adata:
            self.adata = self.load_st_data(lazy=adata_lazy)
        
        if load_wsi:
            self.wsi = self.load_wsi()

    def __repr__(self):
        repr_str = f"HESTSample(sample_id={self.sample_id})\n"
        repr_str += f"  ST file: {self.st_path}\n"
        repr_str += f"  WSI file: {self.wsi_path}\n"
        if self.transcripts_path:
            repr_str += f"  transcripts: {self.transcripts_path}\n"
        if self.patches_dir:
            repr_str += f"  patches dir: {self.patches_dir}\n"
        if self.spatial_plot_path:
            repr_str += f"  spatial plot: {self.spatial_plot_path}\n"
        return repr_str

    # ---------------------------
    # 1) 读取/加载 ST 数据
    # ---------------------------
    def load_st_data(self, lazy: bool = True) -> Optional[anndata.AnnData]:
        """
        加载 ST 数据 (AnnData 对象)。

        参数:
          lazy (bool): 是否使用懒加载模式 (backed='r')。

        返回:
          Optional[anndata.AnnData]: 加载的 AnnData 对象或 None。
        """
        if self.adata is not None:
            return self.adata  # 已加载

        if not os.path.exists(self.st_path):
            print(f"AnnData 文件不存在: {self.st_path}")
            return None

        try:
            if lazy:
                self.adata = sc.read_h5ad(self.st_path, backed='r')
            else:
                self.adata = sc.read_h5ad(self.st_path)
            return self.adata
        except Exception as e:
            print(f"加载 AnnData 失败: {e}")
            return None

    def visualize_comparison(
        self, 
        color: Optional[Union[str, List[str]]] = None,
        use_precomputed_spatial_plot: bool = True
    ):
        """
        创建一个包含 WSI、ST 数据和 QC 数据的综合图像。

        参数:
          color (Optional[Union[str, List[str]]]): Scanpy 可视化的颜色参数。
          use_precomputed_spatial_plot (bool): 是否使用预生成的空间转录组图像。
        """
        if self.adata is None:
            print("AnnData 对象未加载。请确保在初始化时加载或手动加载。")
            return
        
        # 检查是否使用预生成的空间转录组图像
        if use_precomputed_spatial_plot and self.spatial_plot_path and os.path.exists(self.spatial_plot_path):
            spatial_img = Image.open(self.spatial_plot_path)
            use_precomputed = True
        else:
            spatial_img = self.generate_spatial_plot(color=color)
            use_precomputed = False

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # 1. 显示 WSI 缩略图
        if self.wsi:
            thumb = self.get_wsi_thumbnail(level=0, downsample=64)
            axes[0].imshow(thumb)
            axes[0].set_title(f"{self.sample_id} - WSI Thumbnail")
            axes[0].axis('off')
            
            # 在 WSI 图像上添加重要的元数据
            metadata_text = "\n".join([
                f"Sample ID: {self.metadata_dict.get('Sample ID', 'N/A')}",
                f"Organ: {self.metadata_dict.get('organ', 'N/A')}",
                f"Species: {self.metadata_dict.get('species', 'N/A')}",
                f"Disease State: {self.metadata_dict.get('disease_state', 'N/A')}",
                f"Technology: {self.metadata_dict.get('st_technology', 'N/A')}"
            ])
            axes[0].text(10, 30, metadata_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        else:
            axes[0].text(0.5, 0.5, 'WSI not loaded.', 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        fontsize=12)
            axes[0].axis('off')

        # 2. 显示 ST 数据
        if use_precomputed:
            axes[1].imshow(spatial_img)
            axes[1].set_title(f"{self.sample_id} - Spatial Transcriptomics (Precomputed)")
            axes[1].axis('off')
        else:
            sc.pl.spatial(
                self.adata, 
                img=self.get_wsi_thumbnail(level=0, downsample=64),  # 使用缩略图作为背景
                color=color if color else 'clusters',
                show=False,
                ax=axes[1]
            )
            axes[1].set_title(f"{self.sample_id} - Spatial Transcriptomics")
        
        # 3. 显示 QC 数据
        qc_metrics = {
            'Number of Spots Under Tissue': self.metadata_dict.get('Number of Spots Under Tissue', np.nan),
            'Number of Reads': self.metadata_dict.get('Number of Reads', np.nan),
            'Mean Reads per Spot': self.metadata_dict.get('Mean Reads per Spot', np.nan),
            'Valid Barcodes': self.metadata_dict.get('Valid Barcodes', np.nan),
            'Valid UMIs': self.metadata_dict.get('Valid UMIs', np.nan),
            'Sequencing Saturation': self.metadata_dict.get('Sequencing Saturation', np.nan),
            'Fraction of Spots Under Tissue': self.metadata_dict.get('Fraction of Spots Under Tissue', np.nan),
            'Genes Detected': self.metadata_dict.get('Genes Detected', np.nan),
            'Median Genes per Spot': self.metadata_dict.get('Median Genes per Spot', np.nan),
            'Median UMI Counts per Spot': self.metadata_dict.get('Median UMI Counts per Spot', np.nan)
        }

        # 将 QC 数据可视化为条形图
        qc_df = pd.DataFrame(list(qc_metrics.items()), columns=['Metric', 'Value'])
        qc_df.plot(kind='bar', x='Metric', y='Value', ax=axes[2], legend=False)
        axes[2].set_title(f"{self.sample_id} - QC Metrics")
        axes[2].set_ylabel('Value')
        axes[2].tick_params(axis='x', rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    def generate_spatial_plot(self, color: Optional[Union[str, List[str]]] = None) -> Image.Image:
        """
        实时生成空间转录组图像。

        参数:
          color (Optional[Union[str, List[str]]]): Scanpy 可视化的颜色参数。

        返回:
          Image.Image: 生成的空间转录组图像。
        """
        try:
            # 进行基本的预处理
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.highly_variable_genes(self.adata, flavor='seurat', n_top_genes=2000)
            self.adata = self.adata[:, self.adata.var['highly_variable']]
            sc.pp.scale(self.adata, max_value=10)
            sc.tl.pca(self.adata, svd_solver='arpack')
            sc.pp.neighbors(self.adata, n_neighbors=10, n_pcs=40)
            sc.tl.umap(self.adata)
            sc.tl.leiden(self.adata, key_added='clusters')

            # 检查 scalefactors
            scalefactors = self.adata.uns.get('spatial', {}).get('scalefactors', {})
            if scalefactors:
                # 取第一个 scalefactor
                scalefactor_key = next(iter(scalefactors))
                scale_factor = scalefactors[scalefactor_key]
            else:
                print("未找到 scalefactors，使用默认缩放因子 1.0")
                scale_factor = 1.0

            # 使用 Plotly 生成图像并保存为 PNG
            fig = sc.pl.spatial(
                self.adata, 
                img=self.get_wsi_thumbnail(level=0, downsample=64),  # 使用缩略图作为背景
                color=color if color else 'clusters',
                show=False
            )
            fig.savefig("temp_spatial_plot.png")
            spatial_img = Image.open("temp_spatial_plot.png")
            os.remove("temp_spatial_plot.png")
            return spatial_img
        except Exception as e:
            print(f"生成空间转录组图像失败: {e}")
            return Image.new('RGB', (640, 480), color = 'white')

    # ---------------------------
    # 2) 读取/加载 WSI 数据
    # ---------------------------
    def load_wsi(self) -> Optional[openslide.OpenSlide]:
        """
        打开对应的 H&E Whole Slide Image (WSI)。

        返回:
          Optional[openslide.OpenSlide]: 加载的 OpenSlide 对象或 None。
        """
        if self.wsi:
            return self.wsi  # 已加载

        if not os.path.exists(self.wsi_path):
            print(f"WSI 文件不存在: {self.wsi_path}")
            return None

        try:
            self.wsi = openslide.OpenSlide(self.wsi_path)
            return self.wsi
        except Exception as e:
            print(f"加载 WSI 失败: {e}")
            return None

    def get_wsi_thumbnail(self, level: int = 0, downsample: int = 32) -> np.ndarray:
        """
        获取 WSI 的缩略图，用于快速可视化。

        参数:
          level (int): 读取 OpenSlide 的层级 (0 为最高分辨率)。
          downsample (int): 额外的 downsample 因子。

        返回:
          np.ndarray: 缩略图的 NumPy 数组。
        """
        if not self.wsi:
            print("WSI 对象未加载。")
            return np.zeros((100, 100, 3), dtype=np.uint8)  # 返回一个空白图像

        dims = self.wsi.level_dimensions[level]  # (width, height)
        new_size = (dims[0] // downsample, dims[1] // downsample)
        region = self.wsi.read_region((0, 0), level, dims)
        region = region.resize(new_size)
        return np.array(region.convert("RGB"))

    # ---------------------------
    # 3) 读取 Patches / Transcripts 等
    # ---------------------------
    def list_patches(self) -> List[str]:
        """
        列出该样本 patches 文件夹中的文件 (若存在)。

        返回:
          List[str]: patches 文件路径列表。
        """
        if not self.patches_dir or not os.path.isdir(self.patches_dir):
            return []
        return sorted(glob.glob(os.path.join(self.patches_dir, "*.h5")))

    def load_transcripts(self) -> Optional[pd.DataFrame]:
        """
        若是 Xenium 技术或有转录本 parquet 文件，可以从 self.transcripts_path 读取。

        返回:
          Optional[pd.DataFrame]: 转录本数据的 DataFrame 或 None。
        """
        if self.transcripts_path and os.path.exists(self.transcripts_path):
            try:
                return pd.read_parquet(self.transcripts_path)
            except Exception as e:
                print(f"加载 transcripts 失败: {e}")
                return None
        else:
            return None

class HESTDataset:
    """
    管理整个 HEST 数据集，读取全局元数据并根据查询条件筛选样本。
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.metadata_csv = os.path.join(data_dir, "HEST_v1_1_0.csv")
        if not os.path.exists(self.metadata_csv):
            raise FileNotFoundError(f"未找到 metadata CSV: {self.metadata_csv}")

        # 读取 metadata
        self.meta_df = pd.read_csv(self.metadata_csv)
        self.samples_dict = {}  # 存储 sample_id -> HESTSample

    def query_samples(
        self, 
        organ: Optional[str] = None, 
        oncotree_code: Optional[str] = None, 
        sample_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        根据 organ, oncotree_code 或者指定的 sample_ids 在 metadata 里过滤。

        参数:
          organ (Optional[str]): 器官类型。
          oncotree_code (Optional[str]): 癌症类型代码。
          sample_ids (Optional[List[str]]): 特定的样本 ID 列表。

        返回:
          pd.DataFrame: 过滤后的元数据 DataFrame。
        """
        df = self.meta_df.copy()
        if organ:
            df = df[df['organ'] == organ]
        if oncotree_code:
            df = df[df['oncotree_code'] == oncotree_code]
        if sample_ids:
            df = df[df['id'].isin(sample_ids)]
        return df

    def get_samples(
        self, 
        organ: Optional[str] = None, 
        oncotree_code: Optional[str] = None, 
        sample_ids: Optional[List[str]] = None
    ) -> List[HESTSample]:
        """
        返回满足条件的 HESTSample 实例列表。

        参数:
          organ (Optional[str]): 器官类型。
          oncotree_code (Optional[str]): 癌症类型代码。
          sample_ids (Optional[List[str]]): 特定的样本 ID 列表。

        返回:
          List[HESTSample]: 满足条件的样本列表。
        """
        df = self.query_samples(organ, oncotree_code, sample_ids)
        samples = []
        for i, row in df.iterrows():
            sid = row['id']
            # 构造 AnnData 文件路径
            st_path = os.path.join(self.data_dir, "st", f"{sid}.h5ad")
            if not os.path.exists(st_path):
                # 允许更灵活的匹配
                st_candidates = glob.glob(os.path.join(self.data_dir, "st", f"*{sid}*.h5ad"))
                if len(st_candidates) == 0:
                    print(f"未找到 AnnData 文件 for {sid}")
                    continue
                st_path = st_candidates[0]

            # 构造 WSI 文件路径
            wsi_candidates = glob.glob(os.path.join(self.data_dir, "wsis", f"{sid}.*"))
            if len(wsi_candidates) == 0:
                print(f"未找到 WSI 文件 for {sid}")
                wsi_path = ""
            else:
                wsi_path = wsi_candidates[0]  # 取第一个匹配的

            # patches 目录
            patches_dir = os.path.join(self.data_dir, "patches", f"{sid}")
            if not os.path.isdir(patches_dir):
                patches_dir = None

            # transcripts 目录
            transcripts_candidates = glob.glob(os.path.join(self.data_dir, "transcripts", f"{sid}*.parquet"))
            transcripts_path = transcripts_candidates[0] if transcripts_candidates else None

            # spatial_plot 目录
            spatial_plot_path = os.path.join(self.data_dir, "spatial_plots", f"{sid}_spatial.png")
            if not os.path.exists(spatial_plot_path):
                spatial_plot_path = None  # 如果没有预生成的图像

            # 构造 HESTSample 对象
            sample = HESTSample(
                sample_id = sid,
                st_path = st_path,
                wsi_path = wsi_path,
                patches_dir = patches_dir,
                transcripts_path = transcripts_path,
                metadata_dict = row.to_dict(),
                spatial_plot_path = spatial_plot_path,
                load_adata=True,
                adata_lazy=False,  # 根据需要选择懒加载或全加载
                load_wsi=True
            )
            samples.append(sample)
        return samples