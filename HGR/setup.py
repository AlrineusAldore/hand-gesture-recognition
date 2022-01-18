from distutils.core import setup
from Cython.Build import cythonize

extensions = 'cython_funcs\\helpers_cy.pyx'
#ext_options = {"compiler_directives": {"profile": True}, "annotate": True}

extensions = cythonize(extensions, language_level = "3")
setup(ext_modules = extensions)

#from setuptools import setup, Extension
#
#module = Extension ('cython_funcs\\helpers_cy.pyx', sources=['cython_funcs\\helpers_cy.pyx'])
#
#setup(
#    name='HGR',
#    version='1.0',
#    author='me',
#    ext_modules=[module])
