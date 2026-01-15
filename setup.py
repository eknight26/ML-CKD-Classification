from setuptools import setup, find_packages
from typing import List


HYPEN_EDOT = "-e ."
def get_requirements(file_path: str) -> list[str]:

    """This function will return the list of requirements"""

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

        # If there are any specific lines that do not need to be included, handle them here
        # For example, if there's a line for a specific package that should be excluded:
        if HYPEN_EDOT in requirements:
            requirements.remove(HYPEN_EDOT)
    
    return requirements



setup(
    name="Chronic Kidney Disease Prediction ML Project",
    version="0.0.1",
    author="Ernest",
    author_email="ernestcabs@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)