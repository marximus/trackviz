from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='trackviz',
    version='0.1.0',
    description='Visualize trajectories in 2D and 3D using matplotlib',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Maurice Marx',
    author_email='momomarx@gmail.com',
    url='https://github.com/marximus/trackviz',
    project_urls={
        'Bug Reports': 'https://github.com/marximus/trackviz/issues',
        'Source': 'https://github.com/marximus/trackviz',
    },
    license='MIT',
    packages=['trackviz'],
    python_requires='>=3.9',
    install_requires=[
        'numpy>=2.0.0',
        'pandas>=2.2.0',
        'matplotlib>=3.8.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    keywords='trajectory visualization matplotlib tracking 2d 3d',
)
