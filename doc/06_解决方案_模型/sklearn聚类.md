我将帮您使用 sklearn 实现一个股票聚类分析工具，主要使用 K-means 算法，并加入一些评估方法来帮助确定最优聚类数量。



```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

class StockClusterAnalyzer:
    def __init__(self, data, features=None):
        """
        初始化聚类分析器
        
        Parameters:
        data: DataFrame, 包含股票特征的数据框
        features: list, 用于聚类的特征列表
        """
        self.data = data
        self.features = features if features else self.get_default_features()
        self.scaler = StandardScaler()
        self.kmeans = None
        self.scaled_features = None
        
    def get_default_features(self):
        """
        定义默认的特征集
        建议包含：收益率、波动率、换手率、市值等
        """
        return [
            'returns_mean',        # 平均收益率
            'returns_std',         # 收益率标准差（波动率）
            'volume_mean',         # 平均成交量
            'turnover_rate',       # 换手率
            'pe_ratio',           # 市盈率
            'pb_ratio',           # 市净率
            'market_cap'          # 市值
        ]
    
    def prepare_features(self):
        """
        准备特征数据并进行标准化
        """
        # 确保所有特征都存在
        missing_features = [f for f in self.features if f not in self.data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # 提取特征并处理缺失值
        X = self.data[self.features].copy()
        X = X.fillna(X.mean())
        
        # 标准化
        self.scaled_features = self.scaler.fit_transform(X)
        return self.scaled_features
    
    def find_optimal_clusters(self, max_clusters=15):
        """
        使用肘部法则和轮廓系数找到最优聚类数量
        """
        scaled_features = self.prepare_features()
        
        # 计算不同k值的指标
        inertias = []
        silhouette_scores = []
        ch_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))
            ch_scores.append(calinski_harabasz_score(scaled_features, kmeans.labels_))
        
        # 绘制评估指标图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 肘部法则图
        ax1.plot(k_range, inertias, 'bx-')
        ax1.set_xlabel('k')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        
        # 轮廓系数图
        ax2.plot(k_range, silhouette_scores, 'rx-')
        ax2.set_xlabel('k')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        
        # Calinski-Harabasz指数图
        ax3.plot(k_range, ch_scores, 'gx-')
        ax3.set_xlabel('k')
        ax3.set_ylabel('Calinski-Harabasz Score')
        ax3.set_title('Calinski-Harabasz Analysis')
        
        plt.tight_layout()
        plt.show()
        
        # 返回建议的最优聚类数
        return k_range[np.argmax(silhouette_scores)]
    
    def perform_clustering(self, n_clusters=10):
        """
        执行聚类分析
        """
        if self.scaled_features is None:
            self.scaled_features = self.prepare_features()
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans.fit(self.scaled_features)
        
        # 将聚类结果添加到原始数据中
        self.data['cluster'] = self.kmeans.labels_
        return self.kmeans.labels_
    
    def analyze_clusters(self):
        """
        分析聚类结果
        """
        if self.kmeans is None:
            raise ValueError("Please run perform_clustering first")
        
        # 计算每个簇的特征均值
        cluster_means = pd.DataFrame()
        for feature in self.features:
            cluster_means[feature] = [
                self.data[self.data['cluster'] == i][feature].mean()
                for i in range(self.kmeans.n_clusters_)
            ]
        
        # 可视化聚类结果
        # 使用PCA降维到2D进行可视化
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.scaled_features)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                            c=self.kmeans.labels_, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('Cluster Visualization (PCA)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.show()
        
        # 绘制聚类特征热图
        plt.figure(figsize=(12, 8))
        sns.heatmap(cluster_means.T, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Cluster Characteristics')
        plt.xlabel('Cluster')
        plt.ylabel('Features')
        plt.show()
        
        return cluster_means

# 示例用法
def main():
    # 生成示例数据
    np.random.seed(42)
    n_stocks = 5000
    
    sample_data = pd.DataFrame({
        'stock_code': [f'STOCK_{i}' for i in range(n_stocks)],
        'returns_mean': np.random.normal(0.001, 0.002, n_stocks),
        'returns_std': np.random.normal(0.02, 0.005, n_stocks),
        'volume_mean': np.random.normal(1000000, 500000, n_stocks),
        'turnover_rate': np.random.normal(0.02, 0.01, n_stocks),
        'pe_ratio': np.random.normal(15, 5, n_stocks),
        'pb_ratio': np.random.normal(2, 0.5, n_stocks),
        'market_cap': np.random.normal(10000000000, 5000000000, n_stocks)
    })
    
    # 创建分析器实例
    analyzer = StockClusterAnalyzer(sample_data)
    
    # 找到最优聚类数量
    optimal_k = analyzer.find_optimal_clusters()
    print(f"Suggested optimal number of clusters: {optimal_k}")
    
    # 执行聚类
    labels = analyzer.perform_clustering(n_clusters=10)
    
    # 分析聚类结果
    cluster_means = analyzer.analyze_clusters()
    
    # 打印每个簇的大小
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    print("\nCluster sizes:")
    print(cluster_sizes)

if __name__ == "__main__":
    main()

```

这个股票聚类分析工具提供了以下功能：

1. 特征工程：
   - 支持自定义特征选择
   - 包含常用的股票特征（收益率、波动率、换手率等）
   - 自动处理缺失值和标准化

2. 最优聚类数量选择：
   - 使用肘部法则
   - 轮廓系数分析
   - Calinski-Harabasz指数

3. 聚类分析：
   - K-means聚类实现
   - PCA降维可视化
   - 聚类特征分析热图

使用方法：
```python
# 准备数据
data = pd.DataFrame({
    'stock_code': [...],  # 股票代码
    'returns_mean': [...], # 平均收益率
    'returns_std': [...],  # 波动率
    'volume_mean': [...],  # 平均成交量
    'turnover_rate': [...],# 换手率
    'pe_ratio': [...],     # 市盈率
    'pb_ratio': [...],     # 市净率
    'market_cap': [...]    # 市值
})

# 创建分析器实例
analyzer = StockClusterAnalyzer(data)

# 寻找最优聚类数量
optimal_k = analyzer.find_optimal_clusters()

# 执行聚类
labels = analyzer.perform_clustering(n_clusters=10)

# 分析聚类结果
cluster_means = analyzer.analyze_clusters()
```

您可以根据需要调整：

1. 特征选择：修改 `get_default_features()` 方法
2. 聚类数量：修改 `perform_clustering()` 的 `n_clusters` 参数
3. 评估指标：在 `find_optimal_clusters()` 中添加其他评估指标

需要我帮您调整任何参数或添加其他功能吗？