from setuptools import find_packages,setup
from typing import List
hypehn_e = '-e .'
def find_reqs(file_path:str)->List[str]:
    with open(file_path) as fl:
        requremnets = fl.readlines()
        frqs = [rqs.replace('\n','') for rqs in  requremnets]
        if hypehn_e in frqs:
            frqs.remove(hypehn_e)
    return frqs


setup(
    name= "student_performance",
    version = '0.0.1',
    author='adiyann_13',
    packages=find_packages(),
    install_requires = find_reqs('requirements.txt')
)
