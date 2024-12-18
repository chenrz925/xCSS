# 添加源文件
add_library(xcss MODULE project_name.cpp)

# 设置输出名称和 Python 扩展规则
set_target_properties(project_name PROPERTIES
    OUTPUT_NAME "_xcss"  # 对应 Python 中的模块名
    PREFIX ""                   # 防止生成 lib 前缀
    SUFFIX ".so"                # 生成 .so 动态库
)

# 链接 Python 依赖库
target_include_directories(project_name PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(project_name PRIVATE ${Python3_LIBRARIES})
