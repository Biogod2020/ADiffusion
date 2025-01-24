import os
import glob
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import openslide
from typing import Optional, Union, List, Dict
from tqdm import tqdm

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
        metadata_dict: Optional[Dict] = None
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

    def visualize_st_data(
        self, 
        lazy: bool = True, 
        basis: str = 'umap', 
        color: Optional[Union[str, List[str]]] = None
    ):
        """
        利用 scanpy 自带的工具对 ST 数据进行简单的降维可视化
        (示例：UMAP 或 PCA 等).
        需要 full load 才能做基于表达的聚类. 
        如果 lazy=True, 只能做有限的 backed 模式下的操作.
        """
        adata = self.load_st_data(lazy=lazy)
        if lazy:
            print("注意：backed 模式下某些操作(比如高变量基因筛选、聚类)将受限。")

        # 如果是 full 模式，简单演示：pca -> neighbors -> umap
        if not lazy:
            if 'X_pca' not in adata.obsm:
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)
                adata = adata[:, adata.var['highly_variable']]
                sc.pp.scale(adata, max_value=10)
                sc.tl.pca(adata, svd_solver='arpack')
                sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
                sc.tl.umap(adata)
                sc.tl.leiden(adata, key_added='clusters')
            
            sc.pl.umap(adata, color=color if color else 'clusters')
        else:
            # backed 模式下，只能展示已有的 embedding (如果文件里已经有的话)
            if f'X_{basis}' in adata.obsm.keys():
                sc.pl.embedding(
                    adata, 
                    basis=basis, 
                    color=color if color else adata.var_names[0], 
                    title=f"{self.sample_id} - {basis}"
                )
            else:
                print(f"在backed模式下无法自动生成 {basis}。请尝试full模式。")

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

    def get_wsi_thumbnail(self, level: int = 0, downsample: int = 32):
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
                continue

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

            # 构造 sample
            s = HESTSample(
                sample_id = sid,
                st_path = st_path,
                wsi_path = wsi_path,
                patches_dir = patches_dir,
                transcripts_path = transcripts_path,
                metadata_dict = row.to_dict()
            )
            samples.append(s)
        return samples