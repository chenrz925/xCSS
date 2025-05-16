export NERFSTUDIO_METHOD_CONFIGS='xcss=xcss.config:xcss'
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CC=gcc-12
export CXX=g++-12

# ns-render camera-path --load-config outputs/Qianqing-Palace/xcss/2025-04-03_095556/config.yml --camera-path-filename /storage/CSScenes/Qianqing-Palace/camera_paths/2025-04-03-09-57-17.json --output-path renders/Qianqing-Palace/2025-04-03-09-57-17.mp4
ns-render camera-path --load-config outputs/Taj-Falaknuma-Palace/xcss/2025-04-04_085634/config.yml --camera-path-filename /storage/CSScenes/Taj-Falaknuma-Palace/camera_paths/2025-04-04-16-12-26.json --output-path renders/Taj-Falaknuma-Palace/2025-04-04-16-12-26.mp4