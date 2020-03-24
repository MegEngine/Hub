# MegEngine Model Hub

## 如何添加新模型
- 请创建一个新的 Github Repo，放置 `hub_conf.py`
- 在 models 这个目录中放置一个描述文件，文件命名请改为`组织名_模型名.md`
    - 请参考 [样例文件](./megengine_example.md) 的格式
    - 请同时添加中文、英文版本
    - `github_link` 请指向 `hub_conf.py` 所在 repo
- 发一个新的 Pull Request

## 如何测试确认
- 请运行 `scripts/generate_data.py --source=../models --output=/data` 生成 json 文件
