一个收集整洁桌面的数据集

运行环境可以用你之前那个资产收集的那个，依赖见requirements.txt。

运行这个repo还依赖安装https://github.com/ctbfl/organize_it_v2，以确保使用的是同一个asset registry.

将我给你的handcraft bundle解压到tidy_dataset(当前repo)的根目录后，运行

```
python run_constraint_annotation_server.py
```

来打开编辑器。

同时我们仍然在当前repo留了一份资产管理器在`./asset_collection/asset_browser.py`