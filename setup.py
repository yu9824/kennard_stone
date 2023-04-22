# 参考: https://qiita.com/kinpira/items/0a4e7c78fc5dd28bd695

from setuptools import setup, find_packages
import os
import re


# ------------------------ ここを変更 --------------------------------
PACKAGE_NAME = "kennard_stone"  # フォルダの名前も統一
DESCRIPTION = (
    "A method for selecting samples by spreading the training data evenly."
)
# --------------------------------------------------------------------

# read __init__.py
with open(os.path.join(PACKAGE_NAME, "__init__.py"), encoding="utf-8") as f:
    init_text = f.read()
    VERSION = re.search(
        r"__version__\s*=\s*[\'\"](.+?)[\'\"]", init_text
    ).group(1)
    LICENSE = re.search(
        r"__license__\s*=\s*[\'\"](.+?)[\'\"]", init_text
    ).group(1)
    AUTHOR = re.search(r"__author__\s*=\s*[\'\"](.+?)[\'\"]", init_text).group(
        1
    )
    EMAIL = re.search(
        r"__author_email__\s*=\s*[\'\"](.+?)[\'\"]", init_text
    ).group(1)
    ID = re.search(r"__user_id__\s*=\s*[\'\"](.+?)[\'\"]", init_text).group(1)
    APP_NAME = re.search(
        r"__app_name__\s*=\s*[\'\"](.+?)[\'\"]", init_text
    ).group(1)
    url = re.search(r"__url__\s*=\s*[\'\"](.+?)[\'\"]", init_text).group(1)

assert VERSION
assert LICENSE
assert AUTHOR
assert EMAIL
assert ID
assert APP_NAME
assert url

# 参考:
# https://python-packaging-user-guide-ja.readthedocs.io/ja/latest/distributing.html#manifest-in
with open("requirements.txt", encoding="utf-8") as requirements_file:
    install_requirements = requirements_file.read().splitlines()

try:
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
except IOError:
    long_description = ""

setup(
    name=APP_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    # 'text/plain' or 'text/x-rst' or 'text/markdown'
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    maintainer=AUTHOR,
    maintainer_email=EMAIL,
    install_requires=install_requirements,
    url=url,
    # Space-separated keywords for search in PyPI．
    keywords="kennard_stone, scikit-learn, train_test_split, KFold",
    license=LICENSE,
    packages=find_packages(exclude=["example"]),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],  # https://pypi.org/classifiers/
)
