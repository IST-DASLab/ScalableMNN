import setuptools

from torch.utils import cpp_extension


setuptools.setup(
    name='mnn',
    version='1.0.0',
    description='Scalable Mechanistic Neural Networks (ICLR 2025)',
    install_requires=['torch'],
    packages=setuptools.find_packages(exclude=['docs', 'examples', 'tests']),
    ext_modules=[cpp_extension.CppExtension(name='mnn_cpp', sources=['mnn/mnn_cpp.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
