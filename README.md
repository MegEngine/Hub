# MegEngine Model Hub

本仓库包含了[MegEngine模型中心](https://megengine.org.cn/model-hub/)中的模型配置文件，这些模型都可以用`megengine.hub`直接导入。
MegEngine用户在创建了自己的模型后，只需要按照以下步骤添加配置文件，就可以将自己的模型添加到MegEngine的模型中心中，分享给其他人使用。

## 如何添加新模型

- 请创建一个新的Github仓库，仓库中除了你的模型外，还需要在仓库中创建一个名为`hub_conf.py`的文件，`hub_conf.py`的格式可以参考<https://github.com/MegEngine/Models/blob/master/hubconf.py>。
- 请fork本仓库，并在`models`目录中创建一个描述文件，该描述文件不仅包含了你的模型仓库的访问方式，还包括了你的模型的使用说明，具体要求如下：
    - 文件名称规范为`组织名_任务名_模型名.md`
    - 请依照[样例文件](./megengine_example.md)填写描述文件，格式错误可能导致hub解析失败
    - `github_link`指向你的仓库中`hub_conf.py`所在的路径
    - 请同时添加中文、英文版本的文档
- 您可以运行`scripts/generate_data.py --source=../models --output=/your/output/path`生成测试文件（json格式），测试描述文件是否正确。
- 向本仓库提交Pull Request，通过后你的模型就会出现在[MegEngine官网模型中心](https://megengine.org.cn/model-hub/)中，且可以用`megengine.hub`调用。
