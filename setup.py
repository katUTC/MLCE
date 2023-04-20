from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A python package for the analysis of Excel based databases'
LONG_DESCRIPTION = 'A python package for machine learning classification on Excel hosted datasets. Especially designed as a pedagogical tool for biological data, most notably for muscle characterization'

# Setting up
setup(
    name="MLCE",
    version=VERSION,
    author="Katharine Nowakowski",
    author_email="<katharine.nowakowski@utc.fr>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['scipy', 'scikit-learn', 'numpy', 'pandas', 'matplotlib', 'xlsxwriter'],
    keywords=['python', 'ML', 'database', 'biology'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
