from setuptools import setup

setup(
    name='trackviz',
    version='0.1',
    description='Visualize trajectories in 2D and 3D using matplotlib',
    author='Maurice Marx',
    author_email='momomarx@gmail.com',
    license='MIT',
    packages=['trackviz'],
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.1.0',
        'matplotlib>=3.3.0',
    ]
)
