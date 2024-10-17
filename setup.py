import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='hough3d',  
     version='1.0.0',
     author="Jack Featherstone",
     author_email="jack.featherstone@oist.jp",
     license='MIT',
     url='https://github.com/jfeatherstone/hough3d',
     description="Library for detecting lines in a 3D point cloud via Hough transform.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3.11",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
