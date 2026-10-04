# ImageGen 图件来源

当前图1和图3使用内置ImageGen生成的版式/概念素材。图1的实验照片与回复来自原始记录，数值标尺由精确数据绘制。图2、图4、图5和附录总览由真实CSV生成。

- 案例版式：`imagegen_teaser_prompt.txt`；素材 `assets/imagegen_teaser_layout.png`。
- 单栏架构：`imagegen_architecture_print_prompt.txt` 和 `imagegen_architecture_ports_edit.txt`；素材 `assets/imagegen_architecture_print.png`。
- `imagegen_provenance.json`：输入SHA及原照片身份。
- `compose_imagegen_figures.py`：可复算的PDF组合；PNG从当前PDF渲染。

旧布局与被替换的图件不进入当前包。实验照片没有通过图像生成模型重画。
