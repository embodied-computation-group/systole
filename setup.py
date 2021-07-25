import os
import codecs
from setuptools import find_packages, setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()


DESCRIPTION = """Cardiac signal analysis with Python"""

DISTNAME = "systole"
MAINTAINER = "Nicolas Legrand"
MAINTAINER_EMAIL = "nicolas.legrand@cfin.au.dk"
VERSION = "0.2.0a"


if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=open("README.rst").read(),
        long_description_content_type="text/x-rst",
        license="GPL-3.0",
        version=VERSION,
        install_requires=get_requirements(),
        include_package_data=True,
        packages=find_packages(),
    )
