import setuptools

setuptools.setup(
    name="constraint_learning",
    version="0.1dev",
    description="Constraint Learning",
    long_description=open("README.md").read(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pandas",
        "plotnine",
        "sacred",
        "gym",
        "gymnasium==0.28.1",
        "highway-env==1.8.2",
        "cvxopt",
        "einops",
        "scikit-learn",
    ],
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    zip_safe=True,
    entry_points={},
)
