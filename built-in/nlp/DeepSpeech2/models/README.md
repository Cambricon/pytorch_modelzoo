# Ubuntu系统下DeepSpeech2依赖库问题&&解决


## 1. 当确认soundfile已经安装但依然遇到```OSError: sndfile library not found```
```
sudo apt-get install apt-utils libpq-dev libsndfile-dev
```

## 2. 当遇到类似```RuntimeError: Error opening '/tmp/tmp7g16lr7y.wav': File contains data in an unknown format.```
```
sudo apt-get install sox
```

# Centos7系统下DeepSpeech2依赖库问题&&解决

## 1. 当遇到```OSError: cannot load library 'libsndfile.so': libsndfile.so: cannot open shared object file: No such file or directory```
```
yum install libsndfile
```

## 2. 当安装完成libsndfile后，出现```soundfile.LibsndfileError: <exception str() failed>```
```
yum install sox
```

Note: 在Ubuntu系统中安装上述依赖库需要sudo权限
