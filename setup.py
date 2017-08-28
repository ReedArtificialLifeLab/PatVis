from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='patvis',
      version='0.1.0',
      description='A Toolbox for Visualizing Patent Data',
      long_description=readme(),
      keyword='patents gpe artificial life uspto ipc',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.4'
      ],
      url='https://bitbucket.org/reedcac/alife',
      author='Drew Blount, Jacob Menick, Alex Ledger, Zackary Dunivin, and Alec Kosik',
      author_email='akosik7@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['docs', 'tests*']),
      install_requires=[
          'pygraphviz',
          'pymongo',
          'numpy',
          'matplotlib',
          'gensim',
          'sklearn',
          'networkx',
      ],
      setup_requires=[
          'numpy',
      ],
      entry_points={
          'console_scripts' : [ 'pg=patvis.patent_grapher_io:main' ],
      },
      package_data={
      },
      data_files=[],
      include_package_data = True,
      zip_safe=False)
