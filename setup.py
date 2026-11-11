# setup.py
from setuptools import setup, find_packages

setup(
    name="metaseq_analysis",          # 包名
    version="0.1.0",
    author="Sun Mintao",
    description="meta-seq analysis package",
    packages=find_packages(),        # 自动发现 rnaseq_analysis 及子包
    install_requires=[               # 依赖列表（可选）
        "pandas>=1.0",
        "numpy>=1.18",
        "matplotlib>=3.0",
        # 其他依赖...
    ],
    python_requires=">=3.13",
)