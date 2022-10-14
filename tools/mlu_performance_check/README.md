# 性能测试环境检查

## check_mlu_perf.sh

### 1. CheckList
- 检查数据集路径是否挂载。
- 检查NEUWARE_HOME，CATCH_HOME， PYTORCH_HOME， VISION_HOME等环境变量是否设置。
- 检查CATCH依赖的各种MLU neuware库版本是否正确。
- 检查当前linux系统是否为Ubuntu或者Centos操作系统, 并输出对应系统版本号。
- 输出CPU型号，主频，核心数等相关信息。
- 检查CPU是否空闲，如果某个进程的CPU占用率超过5%，则会输出对应进程的PID和和COMMAND。
- 检查MLU是否空闲。
- 检查MLULink，如果发现某张MLU卡有一个链路的status为Disable就会报出对应的卡号。
- 检查MLU task accelerate。
- 检查CPU performance模式。

### 2. 使用方法
> ./check_mlu_perf.sh

## set_mlu_perf.sh

### 1. SetList
- 设置cpu为performance模式。
- 检查MLUTaskAccelerate，并开启MLU的MLUTaskAccelerate。

### 2. 使用方法
> sudo bash set_mlu_perf.sh

## 限制
- DataLoader中的num_works * 使用的卡数小于CPU逻辑核数量。
- 该检查工具仅适用于Centos7, Ubuntu16.04和Ubuntu18.04操作系统。
- 该检查工具仅在Intel CPU上测试过，AMD及其他品牌CPU未进行过测试，若在其他CPU上出现检查进程不准确，请注释掉check_mlu_perf.sh第210行FindCPUProcess函数，这时您需要
  通过top指令检查CPU是否独占。
