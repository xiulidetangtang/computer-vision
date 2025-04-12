# computer-vision
this is for Fudan computer vision class
训练以及测试方法：首先将百度网盘中的模型参数解压缩到与代码同一个目录下面。
在main.py文件中，使用train函数可以进行训练，其中可以调节传入的参数。filepath表示保存参数的地址，epoch是最大epoch数量。后面的超参数均有默认值。
同样可以用test函数进行训练，传入参数文件地址，就可以使用该参数进行测试。
运行export visualize parameters可以输出可视化的参数分布以及隐藏层的热图，我想尝试看隐藏层上是否会有类似于卷积网络一样的特征。
运行grid train函数可以进行网格查找，找到最优参数的位置。find super params函数是用来辅助grid train函数的。
运行grid search.py文件会依次测试所有保存好的参数，并返回一个存储了测试结果的json文件。


