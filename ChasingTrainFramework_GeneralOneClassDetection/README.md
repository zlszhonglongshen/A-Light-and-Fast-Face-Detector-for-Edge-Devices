## ChasingTrainFramework_GeneralSingleClassDetection
本仓库保存了针对单类目标检测(general single-class detection(GSCD))的训练框架代码.主要实现的算法是LFFD.
典型的应用包括 [face detection], [people/pedestrian detection], [head detection], [head&shoulder detection] 等等.
当前版本号: 0.1.0

该框架有如下几个目标:
1, 将配置和训练过程高度自动化,形成流水线
2, 所形成的流水线有标准化的组件和可定制化的组件构成,兼具高效和灵活性
3, 目前框架自带一套高速高精度的检测算法,支持单机多卡训练

框架的重要组成部分及其说明:
[data provider (folder name: data_provider)]
该模块用于是数据供给,它负责将数据从最原始状态转化到可被打包的状态.该模块是可以定制化的.模块内提供了两个基类,一个用于对数据做定制化处理,一个用于数据的打包和数据的读取.

[data iterator (folder name: data_iterator_farm)] 
该模块提供数据迭代器,它负责加载数据,数据预处理,数据打包成batch供训练使用.该模块是可以定制化的.模块内提供了两个数据迭代器.该模块是至关重要的.

[symbol (folder name: symbol_farm)] 
该模块用于定义网络结构,提供训练时可用的symbol,以及部署时可用的symbol.模块内提供了一些默认的网络结构,可供用户直接使用.

[image augmentation (folder name: image_augmentation)] 
该模块用于对图像数据做扩增操作,该模块属于标准模块,基于opencv和numpy来对图像进行各种处理.

[loss layer (folder name: loss_layer_farm)] 
该模块用于定于各种损失函数层,该模块属于标准模块.

[metric (folder name: metric_farm)] 
该模块用于定义如果对网络的阶段输出进行定量评估.该模块可以定制化.模块内提供了一些默认的评估方式可以供用户直接使用.

[speed test (folder name: speed_test)] 
该模块提供了对网络进行测速的类,包括使用cudnn和tensorrt的两个版本.该模块是标准模块.

[]

