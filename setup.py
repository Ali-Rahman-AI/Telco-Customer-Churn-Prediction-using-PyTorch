from setuptools import setup, find_packages

setup(
    name="telco-churn",
    version="0.1.0",
    description="Telco Customer Churn Prediction Model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "torch",
        "torchvision",
    ],
)
