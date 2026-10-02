from setuptools import setup, find_packages

setup(
    name="topocoder",
    version="1.0.0",
    description="Topological Subgraph Condensation and Reasoning for Edge-Constrained Coding Agents",
    author="TopoCoder Research Team",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "networkx>=3.0",
        "numpy>=1.22.0",
        "scipy>=1.9.0",
        "matplotlib>=3.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
        "models": [
            "transformers>=4.40.0",
            "torch>=2.0.0",
            "accelerate>=0.28.0",
        ]
    },
)
