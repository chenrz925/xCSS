import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # 创建构建目录
        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        # 调用 CMake 构建
        subprocess.check_call([
            "cmake", "..",
            f"-DPYTHON_EXECUTABLE={sys.executable}"
        ], cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_temp)

# 调用 setup()
setup(
    name="xcss",
    version="0.1.0",
    packages=["xcss"],
    cmdclass={"build_ext": CMakeBuild},
    ext_modules=[Extension("_xcss", sources=[])],  # 占位符
    zip_safe=False,
)
