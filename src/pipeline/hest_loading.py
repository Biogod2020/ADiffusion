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
    用来表示单个 HEST 样本，包括了:
      - Sample ID (如 'TENX95')
      - 路径到 .h5ad (ST 数据)
      - 路径到 Whole Slide Image
      - 其他可选文件，如 patches, transcripts, 等等
    它同时支持 lazy load 和 full load.
    """

    def __init__(
        self,
        sample_id: str,
        st_path: str,
        wsi_path: str,
        patches_dir: Optional[str] = None,
        transcripts_path: Optional[str] = None,
        metadata_dict: Optional[Dict] = None,
        spatial_plot_path: Optional[str] = None
    ):
        """
        初始化样本对象
        """
        self.sample_id = sample_id
        self.st_path = st_path
        self.wsi_path = wsi_path
        self.patches_dir = patches_dir
        self.transcripts_path = transcripts_path
        self.metadata_dict = metadata_dict if metadata_dict else {}
        self.spatial_plot_path = spatial_plot_path  # 预生成的空间转录组图路径

        self._adata_lazy = None  # 用于lazy加载的Scanpy对象
        self._adata_full = None  # 用于full加载的Scanpy对象
        self._wsi_handle = None  # openslide 句柄

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
    def load_st_data(self, lazy: bool = True) -> anndata.AnnData:
        """
        加载 ST 数据 (scanpy/anndata).
        参数:
          lazy: 是否使用 lazy loading (backed='r') 或者一次性读取到内存.
        返回:
          anndata.AnnData
        """
        if lazy:
            if self._adata_lazy is None:
                # 仅当需要时才生成 backed 模式的 anndata
                self._adata_lazy = sc.read_h5ad(self.st_path, backed='r')
            return self._adata_lazy
        else:
            if self._adata_full is None:
                self._adata_full = sc.read_h5ad(self.st_path)
            return self._adata_full

    def visualize_comparison(
        self, 
        lazy: bool = True, 
        color: Optional[Union[str, List[str]]] = None,
        use_precomputed_spatial_plot: bool = True
    ):
        """
        创建一个包含 WSI、ST 数据和 QC 数据的综合图像。
        参数:
          lazy: 是否使用 lazy loading.
          color: scanpy 可视化的颜色参数。
          use_precomputed_spatial_plot: 是否使用预生成的空间转录组图像.
        """
        adata = self.load_st_data(lazy=lazy)
        if not lazy:
            # 进行基本的预处理
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)
            adata = adata[:, adata.var['highly_variable']]
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, svd_solver='arpack')
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
            sc.tl.umap(adata)
            sc.tl.leiden(adata, key_added='clusters')

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # 1. 显示 WSI 缩略图
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

        # 2. 显示 ST 数据
        if use_precomputed_spatial_plot and self.spatial_plot_path and os.path.exists(self.spatial_plot_path):
            spatial_img = Image.open(self.spatial_plot_path)
            axes[1].imshow(spatial_img)
            axes[1].set_title(f"{self.sample_id} - Spatial Transcriptomics (Precomputed)")
            axes[1].axis('off')
        else:
            sc.pl.spatial(
                adata, 
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

    # ---------------------------
    # 2) 读取/加载 WSI 数据
    # ---------------------------
    def load_wsi(self) -> openslide.OpenSlide:
        """
        打开对应的 H&E Whole Slide Image (WSI).
        """
        if self._wsi_handle is None:
            self._wsi_handle = openslide.OpenSlide(self.wsi_path)
        return self._wsi_handle

    def get_wsi_thumbnail(self, level: int = 0, downsample: int = 32) -> Image.Image:
        """
        获取 WSI 的缩略图，用于快速可视化
        level: 读取 OpenSlide 的层级 (0 为最高分辨率)
        downsample: 额外的 downsample 因子
        返回：PIL.Image
        """
        wsi = self.load_wsi()
        dims = wsi.level_dimensions[level]  # (width, height)
        new_size = (dims[0] // downsample, dims[1] // downsample)
        region = wsi.read_region((0, 0), level, dims)
        return region.resize(new_size)

    # ---------------------------
    # 3) 读取 Patches / Transcripts 等
    # ---------------------------
    def list_patches(self) -> List[str]:
        """
        列出该样本 patches 文件夹中的文件 (若存在).
        """
        if not self.patches_dir or not os.path.isdir(self.patches_dir):
            return []
        return sorted(glob.glob(os.path.join(self.patches_dir, "*.h5")))

    def load_transcripts(self) -> Optional[pd.DataFrame]:
        """
        若是 Xenium 技术或有转录本 parquet 文件，可以从 self.transcripts_path 读取.
        返回: 对应的 DataFrame. (若没有，返回 None)
        """
        if self.transcripts_path and os.path.exists(self.transcripts_path):
            return pd.read_parquet(self.transcripts_path)
        else:
            return None

class HESTDataset:
    """
    整体 HEST 数据集的管理类。它会读取全局 metadata CSV (HEST_v1_1_0.csv)，
    并可根据 organ, oncotree code, 或者指定 ID 来加载若干 Sample.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.metadata_csv = os.path.join(data_dir, "HEST_v1_1_0.csv")
        if not os.path.exists(self.metadata_csv):
            raise FileNotFoundError(f"未找到 metadata CSV: {self.metadata_csv}")

        # 读取 metadata
        self.meta_df = pd.read_csv(self.metadata_csv)
        # 也可以加一些清洗/预处理
        # 根据需要进行
        self.samples_dict = {}  # 存储 sample_id -> HESTSample

    def query_samples(
        self, 
        organ: Optional[str] = None, 
        oncotree_code: Optional[str] = None, 
        sample_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        根据 organ, oncotree_code 或者指定的 sample_ids 在 metadata 里过滤.
        返回: 过滤后的 DataFrame
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
        返回满足条件的 HESTSample 实例列表.
        """
        df = self.query_samples(organ, oncotree_code, sample_ids)
        samples = []
        for i, row in df.iterrows():
            sid = row['id']
            # 构造在本地的路径
            st_path = os.path.join(self.data_dir, "st", f"{sid}.h5ad")
            if not os.path.exists(st_path):
                # 也有可能文件名不是严格 {id}.h5ad，你可以在这里做更灵活的匹配
                # 也可以匹配 "*{sid}*.h5ad" 等
                st_candidates = glob.glob(os.path.join(self.data_dir, "st", f"*{sid}*.h5ad"))
                if len(st_candidates) == 0:
                    continue
                st_path = st_candidates[0]

            # 构造 WSI
            # HEST dataset 中 WSI 文件通常是 wsis/{id}.tif 或者 wsis/{id}.tiff
            # 也有可能是 wsis/{id}.svs，看具体情况
            # 这里示例给出几种可能:
            wsi_candidates = glob.glob(os.path.join(self.data_dir, "wsis", f"{sid}.*"))
            if len(wsi_candidates) == 0:
                # 可能没有找到
                wsi_path = ""
            else:
                wsi_path = wsi_candidates[0]  # 取第一个匹配的

            # patches 目录
            patches_dir = os.path.join(self.data_dir, "patches", f"{sid}")
            if not os.path.isdir(patches_dir):
                patches_dir = None

            # transcripts 目录
            # 仅对 Xenium 等技术才有 transcripts
            transcripts_candidates = glob.glob(os.path.join(self.data_dir, "transcripts", f"{sid}*.parquet"))
            transcripts_path = transcripts_candidates[0] if transcripts_candidates else None

            # spatial_plot 目录
            spatial_plot_path = os.path.join(self.data_dir, "spatial_plots", f"{sid}_spatial.png")
            if not os.path.exists(spatial_plot_path):
                spatial_plot_path = None  # 如果没有预生成的图像

            # 构造 sample
            s = HESTSample(
                sample_id = sid,
                st_path = st_path,
                wsi_path = wsi_path,
                patches_dir = patches_dir,
                transcripts_path = transcripts_path,
                metadata_dict = row.to_dict(),
                spatial_plot_path = spatial_plot_path
            )
            samples.append(s)
        return samples