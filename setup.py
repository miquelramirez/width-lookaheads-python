
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def main():
    extras_require = {
        'dev': ['pytest', 'tox'],
        'test': ['pytest', 'tox'],
    }

    install_requires = [
        'py>=3.5.0',
        'setuptools',
        'numpy',
        'scipy',
        'pandas',
        'gym>=0.10.0'
    ]

    setup(
        name='wizluk',
        version='0.1.0',
        description='wizluk: width-based lookaheads Python library',
        long_description=long_description,
        url='https://github.com/miquelramirez/width-lookaheads-python',
        author="Stefan O'Toole and Miquel Ramirez",
        author_email='-',

        keywords='planning reinforcement-learning gym',
        classifiers=[
            'Development Status :: 3 - Alpha',

            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',

            'License :: MIT License 3)',

            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',

        ],

        # You can just specify package directories manually here if your project is
        # simple. Or you can use find_packages().
        #
        # Alternatively, if you just want to distribute a single Python file, use
        # the `py_modules` argument instead as follows, which will expect a file
        # called `my_module.py` to exist:
        #
        #   py_modules=["my_module"],
        #
        packages=find_packages('src'),  # include all packages under src
        package_dir={'': 'src'},  # tell distutils packages are under src


        # This field lists other packages that your project depends on to run.
        # Any package you put here will be installed by pip when your project is
        # installed, so they must be valid existing projects.
        #
        # For an analysis of "install_requires" vs pip's requirements files see:
        # https://packaging.python.org/en/latest/requirements.html
        install_requires=install_requires,

        # List additional groups of dependencies here (e.g. development
        # dependencies). Users will be able to install these using the "extras"
        # syntax, for example:
        #
        #   $ pip install sampleproject[dev]
        #
        extras_require=extras_require,


        # To provide executable scripts, use entry points in preference to the
        # "scripts" keyword. Entry points provide cross-platform support and allow
        # `pip` to create the appropriate form of executable for the target
        # platform.
        #
        # For example, the following would provide a command called `sample` which
        # executes the function `main` from this package when invoked:
        # entry_points={
        #     'console_scripts': [
        #         'sample=sample:main',
        #     ],
        # },

        # This will include non-code files specified in the manifest, see e.g.
        # http://python-packaging.readthedocs.io/en/latest/non-code-files.html
        include_package_data=True
    )


if __name__ == '__main__':
    main()
